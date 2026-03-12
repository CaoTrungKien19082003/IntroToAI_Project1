import numpy as np
import random

def simulated_annealing_graph(cost_matrix, heuristic, start, goal):
    current  = start
    path     = [current]
    temp     = 100.0
    max_iter = 1000
    for _ in range(max_iter):
        if current == goal:
            break
        neighbors = np.where(cost_matrix[current] > 0)[0]
        if len(neighbors) == 0:
            break
        next_node = np.random.choice(neighbors)
        delta     = heuristic[next_node] - heuristic[current]
        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            current = next_node
            path.append(current)
        temp *= 0.99
    return np.array(path) if path[-1] == goal else np.array([])


def simulated_annealing_continuous(objective, bounds, pop_size=30, max_iter=500):
    """
    Single SA chain with total_iters = pop_size * max_iter evaluations —
    same budget as population-based algos (e.g. PSO/DE/GA).
    History is checkpointed every pop_size steps → len == max_iter+1.
    Cooling: T0=1000 → T≈1 over the full budget.
    """
    n_dims      = bounds.shape[0]
    total_iters = pop_size * max_iter
    cool        = (1e-3) ** (1.0 / total_iters)   # 1000 → ~1 over total_iters steps
    sigma       = 0.1 * (bounds[:, 1] - bounds[:, 0])

    current     = np.random.uniform(bounds[:, 0], bounds[:, 1])
    current_fit = objective(current)
    best        = current.copy()
    best_fit    = current_fit
    temp        = 1000.0

    history   = [best_fit]
    diversity = [0.0]

    for step in range(1, total_iters + 1):
        neighbor = np.clip(current + np.random.normal(0, sigma),
                           bounds[:, 0], bounds[:, 1])
        nfit  = objective(neighbor)
        delta = nfit - current_fit
        if delta < 0 or np.random.rand() < np.exp(-delta / max(temp, 1e-10)):
            current, current_fit = neighbor, nfit
            if nfit < best_fit:
                best, best_fit = neighbor.copy(), nfit
        temp *= cool
        if step % pop_size == 0:
            history.append(best_fit)
            diversity.append(0.0)

    return best, best_fit, history, diversity
