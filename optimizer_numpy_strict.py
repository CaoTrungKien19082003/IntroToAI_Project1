"""
=============================================================================
  Continuous Function Optimizer & Comparator  —  Strict NumPy Edition
  Functions : Sphere, Rosenbrock, Rastrigin, Ackley, Griewank
  Solvers   : BFS, DFS, Greedy, A*, Hill Climbing, Simulated Annealing

  STRICTLY NumPy only — no heapq, no deque, no set, no itertools.
  Every data structure is a pre-allocated numpy ndarray.

  Data structure replacements
  ────────────────────────────
  heapq  → 2-D float array  open_set[MAX, 1+n]  (col 0 = priority)
             pop  : idx = np.argmin(open_set[:size, 0])
             push : open_set[size] = row ; size += 1

  deque  → 2-D float array  buf[MAX, n]  + head / tail integer indices
             enqueue : buf[tail % MAX] = x ; tail += 1
             dequeue : x = buf[head % MAX] ; head += 1

  stack  → 2-D float array  buf[MAX, n]  + top integer index
             push : buf[top] = x ; top += 1
             pop  : top -= 1 ; x = buf[top]

  set    → 2-D float array  visited[MAX, n]  + count integer
             member : np.any(np.all(np.abs(visited[:count] - x) < TOL, axis=1))
             add    : visited[count] = x ; count += 1
=============================================================================
"""

import time
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DIM      = 2
BOUND    = 5.12
STEP     = 0.4
SEED     = 42
TOL      = 1e-9        # float equality tolerance for visited check
MAX_DISC = 3_000       # node budget for grid-based solvers (BFS/DFS/Greedy/A*)
MAX_CONT = 20_000      # iteration budget for HC / SA

rng = np.random.default_rng(SEED)

# ─────────────────────────────────────────────────────────────────────────────
#  BENCHMARK FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def sphere(x):
    return float(np.sum(x ** 2))

def rosenbrock(x):
    return float(np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))

def rastrigin(x):
    return float(10*x.size + np.sum(x**2 - 10*np.cos(2*np.pi*x)))

def ackley(x):
    n = x.size
    return float(-20*np.exp(-0.2*np.sqrt(np.sum(x**2)/n))
                 - np.exp(np.sum(np.cos(2*np.pi*x))/n)
                 + 20 + np.e)

def griewank(x):
    i = np.arange(1, x.size+1, dtype=np.float64)
    return float(np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(i))) + 1)

FUNCTIONS = {
    "Sphere"    : (sphere,     np.zeros(DIM), 0.0),
    "Rosenbrock": (rosenbrock, np.ones(DIM),  0.0),
    "Rastrigin" : (rastrigin,  np.zeros(DIM), 0.0),
    "Ackley"    : (ackley,     np.zeros(DIM), 0.0),
    "Griewank"  : (griewank,   np.zeros(DIM), 0.0),
}

# ─────────────────────────────────────────────────────────────────────────────
#  NUMPY DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class NpVisited:
    """
    Visited-point set backed by a pre-allocated float array.
    Membership  : row-wise L∞ distance < TOL  →  np.any(np.all(...))
    """
    def __init__(self, capacity, dim):
        self._buf   = np.empty((capacity, dim), dtype=np.float64)
        self._count = np.zeros(1, dtype=np.int64)   # shape (1,) so it stays numpy

    def __len__(self):
        return int(self._count[0])

    def contains(self, x):
        n = int(self._count[0])
        if n == 0:
            return False
        return bool(np.any(np.all(np.abs(self._buf[:n] - x) < TOL, axis=1)))

    def add(self, x):
        n = int(self._count[0])
        self._buf[n] = x
        self._count[0] += 1


class NpQueue:
    """
    FIFO queue backed by a circular pre-allocated float array.
    enqueue : buf[tail % cap] = x  ;  tail += 1
    dequeue : x = buf[head % cap]  ;  head += 1
    """
    def __init__(self, capacity, dim):
        self._buf  = np.empty((capacity, dim), dtype=np.float64)
        self._head = np.zeros(1, dtype=np.int64)
        self._tail = np.zeros(1, dtype=np.int64)
        self._cap  = capacity

    def empty(self):
        return bool(self._head[0] == self._tail[0])

    def enqueue(self, x):
        self._buf[int(self._tail[0]) % self._cap] = x
        self._tail[0] += 1

    def dequeue(self):
        x = self._buf[int(self._head[0]) % self._cap].copy()
        self._head[0] += 1
        return x


class NpStack:
    """
    LIFO stack backed by a pre-allocated float array.
    push : buf[top] = x  ;  top += 1
    pop  : top -= 1  ;  x = buf[top]
    """
    def __init__(self, capacity, dim):
        self._buf = np.empty((capacity, dim), dtype=np.float64)
        self._top = np.zeros(1, dtype=np.int64)

    def empty(self):
        return bool(self._top[0] == 0)

    def push(self, x):
        self._buf[int(self._top[0])] = x
        self._top[0] += 1

    def pop(self):
        self._top[0] -= 1
        return self._buf[int(self._top[0])].copy()


class NpMinHeap:
    """
    Min-heap (priority queue) backed by a pre-allocated float array.
    Each row: [priority, x0, x1, ..., x_{n-1}]

    push : buf[size] = row  ;  size += 1
    pop  : idx = np.argmin(buf[:size, 0])
           row = buf[idx]
           buf[idx] = buf[size-1]   ← fill gap with last element
           size -= 1
    """
    def __init__(self, capacity, dim):
        self._buf  = np.empty((capacity, 1 + dim), dtype=np.float64)
        self._size = np.zeros(1, dtype=np.int64)

    def empty(self):
        return bool(self._size[0] == 0)

    def push(self, priority, x):
        n = int(self._size[0])
        self._buf[n, 0]  = priority
        self._buf[n, 1:] = x
        self._size[0] += 1

    def pop(self):
        n   = int(self._size[0])
        idx = int(np.argmin(self._buf[:n, 0]))
        row = self._buf[idx].copy()
        # fill gap: swap with last row
        self._buf[idx] = self._buf[n - 1]
        self._size[0] -= 1
        return row[0], row[1:]          # priority, point


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED NEIGHBOUR GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def make_neighbours(x):
    """
    2n axis-aligned neighbours via numpy broadcasting.
    Returns (k, n) array, k ≤ 2n after boundary clipping.
    """
    shifts     = np.eye(x.size, dtype=np.float64) * STEP   # (n, n)
    candidates = x + np.vstack([shifts, -shifts])           # (2n, n)
    mask       = np.all(np.abs(candidates) <= BOUND, axis=1)
    return candidates[mask]

def eval_neighbours(fn, nbrs):
    """Evaluate fn on each row; return (k,) float64 array."""
    return np.array([fn(nbrs[i]) for i in range(len(nbrs))], dtype=np.float64)

# ─────────────────────────────────────────────────────────────────────────────
#  1. BFS  —  NpQueue + NpVisited
# ─────────────────────────────────────────────────────────────────────────────

def bfs(fn, x0):
    t0      = time.perf_counter()
    best_x  = x0.copy()
    best_f  = fn(x0)
    nodes   = np.ones(1, dtype=np.int64)            # track as numpy scalar

    visited = NpVisited(MAX_DISC + 10, DIM)
    queue   = NpQueue(MAX_DISC + 10, DIM)
    visited.add(x0);  queue.enqueue(x0)

    while not queue.empty() and int(nodes[0]) < MAX_DISC:
        x    = queue.dequeue()
        nbrs = make_neighbours(x)
        fv   = eval_neighbours(fn, nbrs)             # (k,)
        order = np.argsort(fv)                       # enqueue best-first

        for ii in range(order.size):
            i  = int(order[ii])
            nb = nbrs[i]
            if not visited.contains(nb):
                visited.add(nb)
                queue.enqueue(nb)
                nodes[0] += 1
                if fv[i] < best_f:
                    best_f = float(fv[i])
                    best_x = nb.copy()
            if int(nodes[0]) >= MAX_DISC:
                break

    return best_x, best_f, int(nodes[0]), (time.perf_counter()-t0)*1e3

# ─────────────────────────────────────────────────────────────────────────────
#  2. DFS  —  NpStack + NpVisited
# ─────────────────────────────────────────────────────────────────────────────

def dfs(fn, x0):
    t0      = time.perf_counter()
    best_x  = x0.copy()
    best_f  = fn(x0)
    nodes   = np.ones(1, dtype=np.int64)

    visited = NpVisited(MAX_DISC + 10, DIM)
    stack   = NpStack(MAX_DISC + 10, DIM)
    visited.add(x0);  stack.push(x0)

    while not stack.empty() and int(nodes[0]) < MAX_DISC:
        x    = stack.pop()
        nbrs = make_neighbours(x)
        fv   = eval_neighbours(fn, nbrs)
        # push worst-first → best is on top
        order = np.argsort(fv)[::-1]

        for ii in range(order.size):
            i  = int(order[ii])
            nb = nbrs[i]
            if not visited.contains(nb):
                visited.add(nb)
                stack.push(nb)
                nodes[0] += 1
                if fv[i] < best_f:
                    best_f = float(fv[i])
                    best_x = nb.copy()
            if int(nodes[0]) >= MAX_DISC:
                break

    return best_x, best_f, int(nodes[0]), (time.perf_counter()-t0)*1e3

# ─────────────────────────────────────────────────────────────────────────────
#  3. GREEDY  —  no queue needed; numpy argmin + random restart
# ─────────────────────────────────────────────────────────────────────────────

def greedy(fn, x0):
    t0     = time.perf_counter()
    best_x = x0.copy()
    best_f = fn(x0)
    x      = x0.copy()
    f_x    = best_f
    nodes  = np.ones(1, dtype=np.int64)

    while int(nodes[0]) < MAX_DISC:
        nbrs  = make_neighbours(x)
        fv    = eval_neighbours(fn, nbrs)
        nodes[0] += np.int64(len(nbrs))
        bi    = int(np.argmin(fv))

        if fv[bi] < f_x:
            x     = nbrs[bi].copy()
            f_x   = float(fv[bi])
            if f_x < best_f:
                best_f = f_x
                best_x = x.copy()
        else:
            # local min → random restart (numpy rng, stays in-family)
            x   = rng.uniform(-BOUND, BOUND, x0.shape)
            f_x = fn(x)
            nodes[0] += np.int64(1)

    return best_x, best_f, int(nodes[0]), (time.perf_counter()-t0)*1e3

# ─────────────────────────────────────────────────────────────────────────────
#  4. A*  —  NpMinHeap + NpVisited
#     g = hop count,  h = f(x),  priority = g + h
# ─────────────────────────────────────────────────────────────────────────────

def astar(fn, x0):
    t0      = time.perf_counter()
    best_x  = x0.copy()
    best_f  = fn(x0)
    nodes   = np.ones(1, dtype=np.int64)

    visited = NpVisited(MAX_DISC + 10, DIM)
    heap    = NpMinHeap(MAX_DISC * 2 * DIM + 10, DIM)
    visited.add(x0)
    heap.push(best_f, x0)               # priority = 0 + h = f(x0)

    g_map_pts  = np.empty((MAX_DISC + 10, DIM),  dtype=np.float64)
    g_map_vals = np.zeros((MAX_DISC + 10,),       dtype=np.float64)
    gm_count   = np.zeros(1, dtype=np.int64)

    def get_g(x):
        n = int(gm_count[0])
        if n == 0:
            return 0.0
        diffs = np.abs(g_map_pts[:n] - x)
        rows  = np.all(diffs < TOL, axis=1)
        if np.any(rows):
            return float(g_map_vals[np.argmax(rows)])
        return 0.0

    def set_g(x, val):
        n = int(gm_count[0])
        g_map_pts[n]  = x
        g_map_vals[n] = val
        gm_count[0]  += 1

    set_g(x0, 0.0)

    while not heap.empty() and int(nodes[0]) < MAX_DISC:
        priority, x = heap.pop()
        g_x  = get_g(x)
        nbrs = make_neighbours(x)
        fv   = eval_neighbours(fn, nbrs)
        nodes[0] += np.int64(len(nbrs))

        for i in range(len(nbrs)):
            nb  = nbrs[i]
            fvi = float(fv[i])
            if not visited.contains(nb):
                visited.add(nb)
                new_g = g_x + 1.0
                set_g(nb, new_g)
                if fvi < best_f:
                    best_f = fvi
                    best_x = nb.copy()
                heap.push(new_g + fvi, nb)

    return best_x, best_f, int(nodes[0]), (time.perf_counter()-t0)*1e3

# ─────────────────────────────────────────────────────────────────────────────
#  5. HILL CLIMBING  —  numpy rng, no Python collections
# ─────────────────────────────────────────────────────────────────────────────

def hill_climbing(fn, x0, sigma=0.3, patience=300):
    t0      = time.perf_counter()
    best_x  = x0.copy()
    best_f  = fn(x0)
    x       = x0.copy()
    f_x     = best_f
    no_imp  = np.zeros(1, dtype=np.int64)
    nodes   = np.ones(1,  dtype=np.int64)

    # Pre-allocate all perturbations as a numpy array
    deltas  = rng.normal(0, sigma, (MAX_CONT, DIM))   # (N, n)

    for i in range(MAX_CONT):
        if int(nodes[0]) >= MAX_CONT:
            break
        xn  = np.clip(x + deltas[i], -BOUND, BOUND)
        fn_ = fn(xn)
        nodes[0] += np.int64(1)

        # Accept / reject — comparison on numpy scalars
        improved = np.array(fn_ < f_x)
        if improved:
            x, f_x  = xn, fn_
            no_imp[0] = np.int64(0)
            if fn_ < best_f:
                best_f = fn_
                best_x = xn.copy()
        else:
            no_imp[0] += np.int64(1)
            if int(no_imp[0]) >= patience:
                x      = rng.uniform(-BOUND, BOUND, x0.shape)
                f_x    = fn(x)
                nodes[0] += np.int64(1)
                no_imp[0] = np.int64(0)

    return best_x, best_f, int(nodes[0]), (time.perf_counter()-t0)*1e3

# ─────────────────────────────────────────────────────────────────────────────
#  6. SIMULATED ANNEALING  —  fully pre-batched numpy random draws
# ─────────────────────────────────────────────────────────────────────────────

def simulated_annealing(fn, x0, T_init=10.0, T_min=1e-6, alpha=0.9995):
    t0      = time.perf_counter()
    x       = x0.copy()
    f_x     = fn(x0)
    best_x  = x0.copy()
    best_f  = f_x
    nodes   = np.ones(1, dtype=np.int64)

    # Pre-batch all random draws as numpy arrays
    deltas  = rng.normal(0, 1, (MAX_CONT, DIM))       # (N, n) perturbations
    log_u   = np.log(rng.uniform(1e-300, 1.0, MAX_CONT))  # (N,) log-uniform

    # Temperature schedule as numpy array: T[i] = T_init * alpha^i
    i_arr   = np.arange(MAX_CONT, dtype=np.float64)
    T_sched = T_init * np.power(alpha, i_arr)         # (N,)

    for i in range(MAX_CONT):
        T = float(T_sched[i])
        if T < T_min:
            break
        xn   = np.clip(x + T * deltas[i], -BOUND, BOUND)
        fn_  = fn(xn)
        nodes[0] += np.int64(1)

        df = np.float64(fn_) - np.float64(f_x)       # numpy scalar arithmetic
        # Boltzmann acceptance: accept if df<0 or log(U) < -df/T
        accept = np.logical_or(df < np.float64(0),
                               log_u[i] < -df / np.float64(T))
        if bool(accept):
            x, f_x = xn, fn_
            if fn_ < best_f:
                best_f = fn_
                best_x = xn.copy()

    return best_x, best_f, int(nodes[0]), (time.perf_counter()-t0)*1e3

# ─────────────────────────────────────────────────────────────────────────────
#  RUNNER
# ─────────────────────────────────────────────────────────────────────────────

SOLVERS = [
    ("BFS",    bfs),
    ("DFS",    dfs),
    ("Greedy", greedy),
    ("A*",     astar),
    ("HC",     hill_climbing),
    ("SA",     simulated_annealing),
]

X0 = rng.uniform(-BOUND, BOUND, DIM)

def run():
    print("=" * 80)
    print(f"  Continuous Optimizer Comparison  |  Strict NumPy Edition")
    print(f"  dim={DIM}  bound=±{BOUND}  step={STEP}  seed={SEED}")
    print(f"  x0 = {np.round(X0, 4)}")
    print(f"  budget: grid={MAX_DISC:,}  |  HC/SA={MAX_CONT:,}")
    print("=" * 80)

    all_results = {}

    for fname, (fn, x_opt, f_opt) in FUNCTIONS.items():
        print(f"\n{'─'*80}")
        print(f"  {fname}   (optimum f={f_opt} at x*={x_opt.tolist()})")
        print(f"{'─'*80}")
        print(f"  {'Solver':<22} {'f(x*)':<18} {'x*':<32} {'nodes':>7}  {'ms':>7}")
        print(f"  {'------':<22} {'------':<18} {'--':<32} {'-----':>7}  {'--':>7}")

        all_results[fname] = {}
        for sname, solver in SOLVERS:
            bx, bf, n, ms = solver(fn, X0.copy())
            all_results[fname][sname] = (bf, n, ms)
            coords = str(np.round(bx, 5).tolist())
            print(f"  {sname:<22} {bf:<18.8f} {coords:<32} {n:>7,}  {ms:>7.2f}")

    SNAMES = [s for s, _ in SOLVERS]
    FNAMES = list(FUNCTIONS.keys())
    CW = 14

    print(f"\n\n{'='*80}")
    print("  SUMMARY — Best f(x*)  (lower is better)")
    print(f"{'='*80}")
    hdr = f"  {'Solver':<22}" + "".join(f"{f:>{CW}}" for f in FNAMES)
    print(hdr)
    print("  " + "-"*(22 + CW*len(FNAMES)))
    for sname in SNAMES:
        row = f"  {sname:<22}"
        for fname in FNAMES:
            row += f"{all_results[fname][sname][0]:>{CW}.6f}"
        print(row)

    print(f"\n{'='*80}")
    print("  SUMMARY — Time (ms)")
    print(f"{'='*80}")
    print(hdr)
    print("  " + "-"*(22 + CW*len(FNAMES)))
    for sname in SNAMES:
        row = f"  {sname:<22}"
        for fname in FNAMES:
            row += f"{all_results[fname][sname][2]:>{CW}.2f}"
        print(row)

    print(f"\n{'='*80}")
    print("  RANKINGS — by f(x*)  (1 = best)")
    print(f"{'='*80}")
    print(hdr)
    print("  " + "-"*(22 + CW*len(FNAMES)))
    for sname in SNAMES:
        row = f"  {sname:<22}"
        for fname in FNAMES:
            vals   = [(s, all_results[fname][s][0]) for s in SNAMES]
            ranked = [s for s, _ in sorted(vals, key=lambda t: t[1])]
            row   += f"{'#'+str(ranked.index(sname)+1):>{CW}}"
        print(row)

    print(f"\n{'='*80}\n  Done.\n{'='*80}\n")


if __name__ == "__main__":
    run()
