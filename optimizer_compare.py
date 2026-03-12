"""
=============================================================================
  Continuous Function Optimizer & Comparator  —  NumPy Edition
  Functions : Sphere, Rosenbrock, Rastrigin, Ackley, Griewank
  Solvers   : BFS, DFS, Greedy, A*, Hill Climbing, Simulated Annealing
=============================================================================
  Discrete-space adaptation (BFS/DFS/Greedy/A*)
  -----------------------------------------------
  The continuous space [-BOUND, BOUND]^n is navigated on a uniform grid of
  spacing STEP.  From any point x, 2n axis-aligned neighbours are generated
  via numpy broadcasting.  A visited-set (point.tobytes()) prevents revisits.

  Continuous-space solvers (HC / SA)
  ------------------------------------
  HC  : Gaussian perturbation, accept only improvements, random-restart on
        patience expiry.
  SA  : Boltzmann acceptance, temperature-scaled Gaussian steps, geometric
        cooling.  All random draws are pre-batched as numpy arrays.
=============================================================================
"""

import heapq
import time
from collections import deque

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
DIM        = 2
BOUND      = 5.12
STEP       = 0.4        # grid resolution for BFS / DFS / Greedy / A*
SEED       = 42
MAX_DISC   = 5_000      # node budget for grid-based solvers
MAX_CONT   = 20_000     # iteration budget for HC / SA

rng = np.random.default_rng(SEED)

# ─────────────────────────────────────────────────────────────────────────────
#  BENCHMARK FUNCTIONS  (global min = 0 at x=0, except Rosenbrock at x=1)
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
    "Sphere"    : (sphere,     np.zeros(DIM),   0.0),
    "Rosenbrock": (rosenbrock, np.ones(DIM),     0.0),
    "Rastrigin" : (rastrigin,  np.zeros(DIM),   0.0),
    "Ackley"    : (ackley,     np.zeros(DIM),   0.0),
    "Griewank"  : (griewank,   np.zeros(DIM),   0.0),
}

# ─────────────────────────────────────────────────────────────────────────────
#  SHARED UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def make_neighbours(x):
    """
    Generate 2n axis-aligned neighbours via numpy broadcasting.
    Returns array (k, n), k <= 2n after boundary clipping.
    """
    shifts     = np.eye(x.size) * STEP              # (n, n)
    candidates = x + np.vstack([shifts, -shifts])   # (2n, n)
    mask       = np.all(np.abs(candidates) <= BOUND, axis=1)
    return candidates[mask]

def eval_neighbours(fn, nbrs):
    """Evaluate fn on every row of nbrs; return (n,) float array."""
    return np.array([fn(nbrs[i]) for i in range(len(nbrs))], dtype=np.float64)

# ─────────────────────────────────────────────────────────────────────────────
#  1. BFS
# ─────────────────────────────────────────────────────────────────────────────

def bfs(fn, x0):
    t0      = time.perf_counter()
    best_x  = x0.copy();  best_f = fn(x0)
    visited = {x0.tobytes()}
    queue   = deque([x0.copy()])
    nodes   = 1

    while queue and nodes < MAX_DISC:
        x    = queue.popleft()
        nbrs = make_neighbours(x)
        fv   = eval_neighbours(fn, nbrs)
        for i in np.argsort(fv):                     # enqueue best-first
            nb  = nbrs[i];  key = nb.tobytes()
            if key not in visited:
                visited.add(key)
                queue.append(nb.copy())
                nodes += 1
                if fv[i] < best_f:
                    best_f = float(fv[i]);  best_x = nb.copy()
            if nodes >= MAX_DISC:
                break

    return best_x, best_f, nodes, (time.perf_counter()-t0)*1e3

# ─────────────────────────────────────────────────────────────────────────────
#  2. DFS
# ─────────────────────────────────────────────────────────────────────────────

def dfs(fn, x0):
    t0      = time.perf_counter()
    best_x  = x0.copy();  best_f = fn(x0)
    visited = {x0.tobytes()}
    stack   = [x0.copy()]
    nodes   = 1

    while stack and nodes < MAX_DISC:
        x    = stack.pop()
        nbrs = make_neighbours(x)
        fv   = eval_neighbours(fn, nbrs)
        for i in np.argsort(fv)[::-1]:              # push worst-first → best popped first
            nb  = nbrs[i];  key = nb.tobytes()
            if key not in visited:
                visited.add(key)
                stack.append(nb.copy())
                nodes += 1
                if fv[i] < best_f:
                    best_f = float(fv[i]);  best_x = nb.copy()
            if nodes >= MAX_DISC:
                break

    return best_x, best_f, nodes, (time.perf_counter()-t0)*1e3

# ─────────────────────────────────────────────────────────────────────────────
#  3. GREEDY  (steepest-descent on grid + random restart)
# ─────────────────────────────────────────────────────────────────────────────

def greedy(fn, x0):
    t0     = time.perf_counter()
    best_x = x0.copy();  best_f = fn(x0)
    x      = x0.copy();  f_x    = best_f
    nodes  = 1

    while nodes < MAX_DISC:
        nbrs = make_neighbours(x)
        fv   = eval_neighbours(fn, nbrs)
        nodes += len(nbrs)
        bi   = int(np.argmin(fv))
        if fv[bi] < f_x:
            x, f_x = nbrs[bi].copy(), float(fv[bi])
            if f_x < best_f:
                best_f = f_x;  best_x = x.copy()
        else:
            # local min → random restart
            x   = rng.uniform(-BOUND, BOUND, x0.shape)
            f_x = fn(x);  nodes += 1

    return best_x, best_f, nodes, (time.perf_counter()-t0)*1e3

# ─────────────────────────────────────────────────────────────────────────────
#  4. A*   g = hop count,  h = f(x),  priority = g + h
# ─────────────────────────────────────────────────────────────────────────────

def astar(fn, x0):
    t0      = time.perf_counter()
    best_x  = x0.copy();  best_f = fn(x0)
    visited = {x0.tobytes()}
    heap    = [(best_f, 0, x0.tobytes())]            # (priority, g, bytes)
    nodes   = 1

    while heap and nodes < MAX_DISC:
        _, g, xb = heapq.heappop(heap)
        x    = np.frombuffer(xb, dtype=np.float64).copy()
        nbrs = make_neighbours(x)
        fv   = eval_neighbours(fn, nbrs)
        nodes += len(nbrs)

        for i in range(len(nbrs)):
            nb  = nbrs[i];  key = nb.tobytes()
            if key not in visited:
                visited.add(key)
                fvi = float(fv[i])
                if fvi < best_f:
                    best_f = fvi;  best_x = nb.copy()
                heapq.heappush(heap, (g+1 + fvi, g+1, key))

    return best_x, best_f, nodes, (time.perf_counter()-t0)*1e3

# ─────────────────────────────────────────────────────────────────────────────
#  5. HILL CLIMBING  (stochastic, Gaussian steps, random restart)
# ─────────────────────────────────────────────────────────────────────────────

def hill_climbing(fn, x0, sigma=0.3, patience=300):
    t0      = time.perf_counter()
    best_x  = x0.copy();  best_f = fn(x0)
    x       = x0.copy();  f_x    = best_f
    no_imp  = 0;  nodes = 1

    while nodes < MAX_CONT:
        delta = rng.normal(0, sigma, x.shape)
        xn    = np.clip(x + delta, -BOUND, BOUND)
        fn_   = fn(xn);  nodes += 1
        if fn_ < f_x:
            x, f_x  = xn, fn_
            no_imp  = 0
            if fn_ < best_f:
                best_f = fn_;  best_x = xn.copy()
        else:
            no_imp += 1
            if no_imp >= patience:
                x      = rng.uniform(-BOUND, BOUND, x0.shape)
                f_x    = fn(x);  nodes += 1
                no_imp = 0

    return best_x, best_f, nodes, (time.perf_counter()-t0)*1e3

# ─────────────────────────────────────────────────────────────────────────────
#  6. SIMULATED ANNEALING
# ─────────────────────────────────────────────────────────────────────────────

def simulated_annealing(fn, x0, T_init=10.0, T_min=1e-6, alpha=0.9995):
    t0      = time.perf_counter()
    x       = x0.copy();  f_x = fn(x0)
    best_x  = x0.copy();  best_f = f_x
    T       = T_init;  nodes = 1

    # pre-batch random draws for speed
    deltas  = rng.normal(0, 1, (MAX_CONT, x.size))
    log_u   = np.log(rng.uniform(1e-300, 1.0, MAX_CONT))

    for i in range(MAX_CONT):
        if T < T_min:
            break
        xn   = np.clip(x + T*deltas[i], -BOUND, BOUND)
        fn_  = fn(xn);  nodes += 1
        df   = fn_ - f_x
        if df < 0 or log_u[i] < -df/T:
            x, f_x = xn, fn_
            if fn_ < best_f:
                best_f = fn_;  best_x = xn.copy()
        T *= alpha

    return best_x, best_f, nodes, (time.perf_counter()-t0)*1e3

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

# Fixed starting point (same for all solvers and functions)
X0 = rng.uniform(-BOUND, BOUND, DIM)

def run():
    print("=" * 78)
    print(f"  Continuous Optimizer Comparison | dim={DIM} | bound=±{BOUND} | seed={SEED}")
    print(f"  x0 = {np.round(X0, 4)}")
    print(f"  Grid step={STEP} | budget: grid={MAX_DISC:,} | HC/SA={MAX_CONT:,}")
    print("=" * 78)

    # ── Per-function detail ────────────────────────────────────────────────
    all_results = {}   # {fname: {sname: (x*, f*, nodes, ms)}}

    for fname, (fn, x_opt, f_opt) in FUNCTIONS.items():
        print(f"\n{'─'*78}")
        print(f"  {fname}   (known optimum: f={f_opt}  at x*={x_opt[:4].tolist()})")
        print(f"{'─'*78}")
        print(f"  {'Solver':<22} {'f(x*)':<16} {'x*':<36} {'nodes':>7}  {'ms':>7}")
        print(f"  {'------':<22} {'------':<16} {'--':<36} {'-----':>7}  {'--':>7}")

        all_results[fname] = {}
        for sname, solver in SOLVERS:
            bx, bf, n, ms = solver(fn, X0.copy())
            all_results[fname][sname] = (bf, n, ms)
            coords = str(np.round(bx, 5).tolist())
            print(f"  {sname:<22} {bf:<16.8f} {coords:<36} {n:>7,}  {ms:>7.2f}")

    # ── Summary: f(x*) comparison table ───────────────────────────────────
    SNAMES = [s for s, _ in SOLVERS]
    FNAMES = list(FUNCTIONS.keys())
    CW = 14   # column width

    print(f"\n\n{'='*78}")
    print("  SUMMARY — Best f(x*) (lower is better)")
    print(f"{'='*78}")
    header = f"  {'Solver':<22}" + "".join(f"{f:>{CW}}" for f in FNAMES)
    print(header)
    print("  " + "-"*(22 + CW*len(FNAMES)))
    for sname in SNAMES:
        row = f"  {sname:<22}"
        for fname in FNAMES:
            row += f"{all_results[fname][sname][0]:>{CW}.6f}"
        print(row)

    # ── Summary: time comparison table ────────────────────────────────────
    print(f"\n{'='*78}")
    print("  SUMMARY — Time (ms)")
    print(f"{'='*78}")
    print(header)
    print("  " + "-"*(22 + CW*len(FNAMES)))
    for sname in SNAMES:
        row = f"  {sname:<22}"
        for fname in FNAMES:
            row += f"{all_results[fname][sname][2]:>{CW}.2f}"
        print(row)

    # ── Rankings per function ──────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("  RANKINGS — by f(x*)  (1 = best)")
    print(f"{'='*78}")
    print(header)
    print("  " + "-"*(22 + CW*len(FNAMES)))
    for sname in SNAMES:
        row = f"  {sname:<22}"
        for fname in FNAMES:
            vals   = [(s, all_results[fname][s][0]) for s in SNAMES]
            ranked = [s for s, _ in sorted(vals, key=lambda t: t[1])]
            rank   = ranked.index(sname) + 1
            row   += f"{'#'+str(rank):>{CW}}"
        print(row)

    print(f"\n{'='*78}")
    print("  Done.")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    run()
