"""
continuous_classic_algorithms.py
─────────────────────────────────────────────────────────────────────────────
BFS, DFS, Greedy, A*, Hill Climbing, and Simulated Annealing adapted to run
on continuous benchmark functions (Sphere, Rosenbrock, Rastrigin, Ackley,
Griewank).

Every function has the SAME signature and return format as META_ALGORITHMS:

    algo(objective, bounds, pop_size, max_iter)
        -> (best_pos, best_fit, history, diversity)

    objective   callable(x: ndarray) -> float   (minimise)
    bounds      ndarray shape (dim, 2)
    pop_size    int   used as evaluation-budget chunk size
    max_iter    int   number of history checkpoints to record
    returns     best_pos, best_fit,
                history  list[float]  length max_iter+1
                diversity list[float] length max_iter+1  (0 for single-point methods)

Grid search methods  (BFS / DFS / Greedy / A*)
──────────────────────────────────────────────
The continuous domain is sampled on a uniform axis-aligned grid with step
  step = mean_range / GRID_DIV
From every point x the 2·dim axis-aligned neighbours are generated via
numpy broadcasting.  A Python set of .tobytes() keys (rounded to the grid)
provides O(1) visited deduplication — no heapq or deque is used for
frontier data structures; those are backed by pre-allocated numpy arrays.
Total evaluations = pop_size * max_iter, capped at MAX_GRID_NODES.

Continuous methods  (HC / SA)
──────────────────────────────
Gaussian-perturbation in the original continuous space.
All random draws are pre-batched as numpy arrays before the main loop.
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────
GRID_DIV       = 12     # step = mean_range / GRID_DIV
MAX_GRID_NODES = 5_000  # hard cap for BFS / DFS / A* node budget
HC_SIGMA_FRAC  = 0.08   # HC sigma = frac * mean_range
HC_PATIENCE    = 250    # HC restarts after this many non-improving steps
SA_T_FRAC      = 0.30   # SA T_init = frac * mean_range
SA_ALPHA       = 0.995 # SA geometric cooling rate


# ─────────────────────────────────────────────────────────────────────────────
#  Strict-NumPy frontier data structures
# ─────────────────────────────────────────────────────────────────────────────

class _NpQueue:
    """FIFO circular queue — float64 array [cap, dim]."""
    def __init__(self, cap, dim):
        self._buf  = np.empty((cap, dim), dtype=np.float64)
        self._head = np.int64(0)
        self._tail = np.int64(0)
        self._cap  = np.int64(cap)

    def empty(self):  return self._head == self._tail
    def enqueue(self, x):
        self._buf[int(self._tail % self._cap)] = x
        self._tail += np.int64(1)
    def dequeue(self):
        row = self._buf[int(self._head % self._cap)].copy()
        self._head += np.int64(1)
        return row


class _NpStack:
    """LIFO stack — float64 array [cap, dim]."""
    def __init__(self, cap, dim):
        self._buf = np.empty((cap, dim), dtype=np.float64)
        self._top = np.int64(0)

    def empty(self):  return self._top == np.int64(0)
    def push(self, x):
        self._buf[int(self._top)] = x
        self._top += np.int64(1)
    def pop(self):
        self._top -= np.int64(1)
        return self._buf[int(self._top)].copy()


class _NpMinHeap:
    """Min-heap — float64 array [cap, 1+dim], col-0 = priority."""
    def __init__(self, cap, dim):
        self._buf  = np.empty((cap, 1 + dim), dtype=np.float64)
        self._size = np.int64(0)

    def empty(self):  return self._size == np.int64(0)
    def push(self, priority, x):
        n = int(self._size)
        self._buf[n, 0]  = np.float64(priority)
        self._buf[n, 1:] = x
        self._size += np.int64(1)
    def pop(self):
        n   = int(self._size)
        idx = int(np.argmin(self._buf[:n, 0]))
        row = self._buf[idx].copy()
        self._buf[idx] = self._buf[n - 1]
        self._size -= np.int64(1)
        return float(row[0]), row[1:]   # priority, point


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _grid_setup(bounds):
    """Return (dim, lower, upper, step) from a (dim, 2) bounds array."""
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    step  = float(np.mean(upper - lower)) / GRID_DIV
    return bounds.shape[0], lower, upper, step


def _neighbours(x, step, lower, upper):
    """
    Generate 2·dim axis-aligned neighbours via numpy broadcasting.
    Returns a (k, dim) array, k ≤ 2·dim after boundary enforcement.
    """
    shifts     = np.eye(x.size, dtype=np.float64) * step
    candidates = x + np.vstack([shifts, -shifts])          # (2·dim, dim)
    mask = np.all((candidates >= lower) & (candidates <= upper), axis=1)
    return candidates[mask]


def _key(x, step):
    """Round x to grid and return bytes — O(1) hash key for visited set."""
    return np.round(x / step).astype(np.int32).tobytes()


def _start_point(bounds):
    """Use the centre of the search space as the starting point."""
    return np.clip(np.zeros(bounds.shape[0]), bounds[:, 0], bounds[:, 1])


def _build_history(best_f_arr, max_iter):
    """
    Subsample a flat best-fitness array into exactly max_iter+1 checkpoints.
    best_f_arr[i] = best fitness after evaluation i.
    """
    n = len(best_f_arr)
    if n == 0:
        return [np.inf] * (max_iter + 1)
    idx = np.round(np.linspace(0, n - 1, max_iter + 1)).astype(int)
    return [float(best_f_arr[i]) for i in idx]


# ─────────────────────────────────────────────────────────────────────────────
#  1.  BFS  (level-order, best-neighbour-first enqueue)
# ─────────────────────────────────────────────────────────────────────────────

def bfs_continuous(objective, bounds, pop_size, max_iter):
    """Level-order grid BFS.  Enqueues neighbours in ascending f order."""
    dim, lower, upper, step = _grid_setup(bounds)
    budget  = min(pop_size * max_iter, MAX_GRID_NODES)
    cap     = budget * 2 * dim + 64

    x0      = _start_point(bounds)
    best_x  = x0.copy()
    best_f  = objective(x0)
    track   = [best_f]              # one entry per evaluation

    visited = {_key(x0, step)}
    q       = _NpQueue(cap, dim)
    q.enqueue(x0)
    evals   = np.int64(1)

    while not q.empty() and int(evals) < budget:
        x    = q.dequeue()
        nbrs = _neighbours(x, step, lower, upper)
        if nbrs.size == 0:
            continue
        fv    = np.array([objective(nbrs[i]) for i in range(len(nbrs))])
        evals += np.int64(len(nbrs))
        order  = np.argsort(fv)          # enqueue cheapest first

        for ii in order:
            nb = nbrs[int(ii)]
            k  = _key(nb, step)
            if k not in visited:
                visited.add(k)
                q.enqueue(nb)
                if float(fv[int(ii)]) < best_f:
                    best_f = float(fv[int(ii)])
                    best_x = nb.copy()
            track.append(best_f)
            if int(evals) >= budget:
                break

    return best_x, best_f, _build_history(track, max_iter), [0.0]*(max_iter+1)


# ─────────────────────────────────────────────────────────────────────────────
#  2.  DFS  (stack, greedy-depth: pushes worst first so best is popped first)
# ─────────────────────────────────────────────────────────────────────────────

def dfs_continuous(objective, bounds, pop_size, max_iter):
    """Depth-first grid DFS.  Best neighbour explored first (LIFO order)."""
    dim, lower, upper, step = _grid_setup(bounds)
    budget  = min(pop_size * max_iter, MAX_GRID_NODES)
    cap     = budget * 2 * dim + 64

    x0      = _start_point(bounds)
    best_x  = x0.copy()
    best_f  = objective(x0)
    track   = [best_f]

    visited = {_key(x0, step)}
    stk     = _NpStack(cap, dim)
    stk.push(x0)
    evals   = np.int64(1)

    while not stk.empty() and int(evals) < budget:
        x    = stk.pop()
        nbrs = _neighbours(x, step, lower, upper)
        if nbrs.size == 0:
            continue
        fv    = np.array([objective(nbrs[i]) for i in range(len(nbrs))])
        evals += np.int64(len(nbrs))
        order  = np.argsort(fv)[::-1]   # push worst first → best popped first

        for ii in order:
            nb = nbrs[int(ii)]
            k  = _key(nb, step)
            if k not in visited:
                visited.add(k)
                stk.push(nb)
                if float(fv[int(ii)]) < best_f:
                    best_f = float(fv[int(ii)])
                    best_x = nb.copy()
            track.append(best_f)
            if int(evals) >= budget:
                break

    return best_x, best_f, _build_history(track, max_iter), [0.0]*(max_iter+1)


# ─────────────────────────────────────────────────────────────────────────────
#  3.  Greedy steepest-descent + random restart
# ─────────────────────────────────────────────────────────────────────────────

def greedy_continuous(objective, bounds, pop_size, max_iter):
    """
    Always step to the best neighbour; random-restart when stuck at a local min.
    No visited set needed — restarts escape repeated visits naturally.
    """
    dim, lower, upper, step = _grid_setup(bounds)
    budget = pop_size * max_iter
    rng    = np.random.default_rng()

    x0     = _start_point(bounds)
    best_x = x0.copy()
    best_f = objective(x0)
    x, f_x = x0.copy(), best_f
    track  = [best_f]
    evals  = np.int64(1)

    while int(evals) < budget:
        nbrs = _neighbours(x, step, lower, upper)
        fv   = np.array([objective(nbrs[i]) for i in range(len(nbrs))])
        evals += np.int64(len(nbrs))
        bi   = int(np.argmin(fv))

        if float(fv[bi]) < f_x:          # improvement → move
            x, f_x = nbrs[bi].copy(), float(fv[bi])
            if f_x < best_f:
                best_f, best_x = f_x, x.copy()
        else:                              # stuck → random restart
            x   = rng.uniform(lower, upper)
            f_x = objective(x)
            evals += np.int64(1)
            if f_x < best_f:
                best_f, best_x = f_x, x.copy()

        track.append(best_f)

    return best_x, best_f, _build_history(track, max_iter), [0.0]*(max_iter+1)


# ─────────────────────────────────────────────────────────────────────────────
#  4.  A*  (priority = g_hop + f(x))
# ─────────────────────────────────────────────────────────────────────────────

def astar_continuous(objective, bounds, pop_size, max_iter):
    """
    A* on grid.  g = hop count from start, h = f(x).
    Balances breadth (low g) with quality (low h).
    """
    dim, lower, upper, step = _grid_setup(bounds)
    budget  = min(pop_size * max_iter, MAX_GRID_NODES)
    cap     = budget * 2 * dim + 64

    x0      = _start_point(bounds)
    best_x  = x0.copy()
    best_f  = objective(x0)
    h0      = best_f
    track   = [best_f]

    visited = {_key(x0, step)}
    heap    = _NpMinHeap(cap, dim)
    heap.push(float(h0), x0)
    g_map   = {_key(x0, step): 0.0}      # key → g value
    evals   = np.int64(1)

    while not heap.empty() and int(evals) < budget:
        priority, x = heap.pop()
        xk   = _key(x, step)
        g_x  = g_map.get(xk, 0.0)

        nbrs = _neighbours(x, step, lower, upper)
        if nbrs.size == 0:
            continue
        fv    = np.array([objective(nbrs[i]) for i in range(len(nbrs))])
        evals += np.int64(len(nbrs))

        for i in range(len(nbrs)):
            nb = nbrs[i]
            nk = _key(nb, step)
            fi = float(fv[i])
            if nk not in visited:
                visited.add(nk)
                new_g      = g_x + 1.0
                g_map[nk]  = new_g
                if fi < best_f:
                    best_f, best_x = fi, nb.copy()
                heap.push(new_g + fi, nb)
            track.append(best_f)
            if int(evals) >= budget:
                break

    return best_x, best_f, _build_history(track, max_iter), [0.0]*(max_iter+1)


# ─────────────────────────────────────────────────────────────────────────────
#  5.  UCS  (Uniform Cost Search — priority = cumulative Euclidean path cost)
# ─────────────────────────────────────────────────────────────────────────────

def ucs_continuous(objective, bounds, pop_size, max_iter):
    """
    Uniform Cost Search on the continuous grid.

    Priority = g  where  g = cumulative Euclidean distance travelled from
    the start point.  There is no heuristic term, so the frontier is ordered
    purely by travel cost — nodes reachable cheaply (close to the start) are
    expanded first.

    How this differs from the other grid methods
    ─────────────────────────────────────────────
    BFS     : ignores edge cost entirely (level order)
    Greedy  : priority = f(x)              (no path cost)
    A*      : priority = g_hops + f(x)     (hop count + function value)
    UCS     : priority = g_euclidean       (Euclidean distance from start, no f)

    On a uniform grid every step has the same Euclidean length (= step), so
    UCS expands in concentric "shells" of equal travel distance — identical
    to BFS in terms of *order* but it tracks real distance and can be updated
    if edge costs ever vary.  The key distinction is that UCS records best
    f(x) seen across ALL expanded nodes without using f as a priority signal,
    making it a pure cost-blind exhaustive sweep.
    """
    dim, lower, upper, step = _grid_setup(bounds)
    budget  = min(pop_size * max_iter, MAX_GRID_NODES)
    cap     = budget * 2 * dim + 64

    x0      = _start_point(bounds)
    best_x  = x0.copy()
    best_f  = objective(x0)
    track   = [best_f]

    visited = {_key(x0, step)}
    heap    = _NpMinHeap(cap, dim)
    heap.push(0.0, x0)                    # initial g = 0
    g_map   = {_key(x0, step): 0.0}      # key → best known g
    evals   = np.int64(1)

    while not heap.empty() and int(evals) < budget:
        g_x, x = heap.pop()
        xk = _key(x, step)

        # Lazy deletion: skip if we already found a cheaper path to this node
        if g_x > g_map.get(xk, np.inf) + 1e-12:
            continue

        nbrs = _neighbours(x, step, lower, upper)
        if nbrs.size == 0:
            continue
        fv    = np.array([objective(nbrs[i]) for i in range(len(nbrs))])
        evals += np.int64(len(nbrs))

        for i in range(len(nbrs)):
            nb   = nbrs[i]
            nk   = _key(nb, step)
            fi   = float(fv[i])
            # Edge cost = Euclidean distance between grid neighbours = step
            new_g = g_x + float(np.linalg.norm(nb - x))   # always == step

            if nk not in visited or new_g < g_map.get(nk, np.inf) - 1e-12:
                visited.add(nk)
                g_map[nk] = new_g
                heap.push(new_g, nb)     # priority = cumulative travel cost
                if fi < best_f:
                    best_f, best_x = fi, nb.copy()

            track.append(best_f)
            if int(evals) >= budget:
                break

    return best_x, best_f, _build_history(track, max_iter), [0.0]*(max_iter+1)


# ─────────────────────────────────────────────────────────────────────────────
#  7.  Hill Climbing  (Gaussian perturbation, random restarts)
# ─────────────────────────────────────────────────────────────────────────────

def hc_continuous(objective, bounds, pop_size, max_iter):
    """
    Stochastic HC with Gaussian perturbation.
    Accepts only improvements; random-restarts after HC_PATIENCE failures.
    All random draws pre-batched as numpy arrays.
    """
    dim, lower, upper, step = _grid_setup(bounds)
    budget = pop_size * max_iter
    sigma  = HC_SIGMA_FRAC * float(np.mean(upper - lower))
    rng    = np.random.default_rng()

    x0     = _start_point(bounds)
    best_x = x0.copy()
    best_f = objective(x0)
    x, f_x = x0.copy(), best_f
    track  = [best_f]

    # Pre-batch all random draws as numpy arrays
    deltas  = rng.normal(0.0, sigma, (budget, dim))
    no_imp  = 0

    for i in range(budget):
        xn  = np.clip(x + deltas[i], lower, upper)
        fn  = objective(xn)

        if fn < f_x:
            x, f_x = xn, fn
            no_imp = 0
            if fn < best_f:
                best_f, best_x = fn, xn.copy()
        else:
            no_imp += 1
            if no_imp >= HC_PATIENCE:
                x   = rng.uniform(lower, upper)
                f_x = objective(x)
                no_imp = 0
                if f_x < best_f:
                    best_f, best_x = f_x, x.copy()

        track.append(best_f)

    return best_x, best_f, _build_history(track, max_iter), [0.0]*(max_iter+1)


# ─────────────────────────────────────────────────────────────────────────────
#  8.  Simulated Annealing  (Gaussian perturbation, geometric cooling)
# ─────────────────────────────────────────────────────────────────────────────

def sa_continuous(objective, bounds, pop_size, max_iter):
    """
    SA with Gaussian perturbation.
    Full temperature schedule and all random draws pre-computed as numpy arrays.
    """
    dim, lower, upper, step = _grid_setup(bounds)
    budget = pop_size * max_iter
    T_init = SA_T_FRAC * float(np.mean(upper - lower))
    rng    = np.random.default_rng()

    x0     = _start_point(bounds)
    best_x = x0.copy()
    best_f = objective(x0)
    x, f_x = x0.copy(), best_f
    track  = [best_f]

    # Pre-batch everything as numpy arrays
    i_arr   = np.arange(budget, dtype=np.float64)
    T_sched = np.float64(T_init) * np.power(np.float64(SA_ALPHA), i_arr)  # (budget,)
    sigma   = float(np.mean(upper - lower)) * 0.1
    deltas  = rng.normal(0.0, sigma, (budget, dim))                        # (budget, dim)
    log_u   = np.log(rng.uniform(1e-300, 1.0, budget))                    # (budget,)

    for i in range(budget):
        T = float(T_sched[i])
        if T < 1e-10:
            break

        xn  = np.clip(x + deltas[i], lower, upper)
        fn  = objective(xn)
        df  = np.float64(fn) - np.float64(f_x)

        # Boltzmann acceptance via numpy scalar ops (no Python math.exp)
        accept = bool(np.logical_or(df < np.float64(0.0),
                                    log_u[i] < -df / np.float64(T)))
        if accept:
            x, f_x = xn, fn
            if fn < best_f:
                best_f, best_x = fn, xn.copy()

        track.append(best_f)

    return best_x, best_f, _build_history(track, max_iter), [0.0]*(max_iter+1)
