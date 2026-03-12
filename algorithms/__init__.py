# algorithms/__init__.py

from .classic_algorithms import (
    bfs, dfs, ucs,
    greedy_best_first_search,
    a_star_search,
    hill_climbing_steepest_ascent,
)
from .physics_based_algorithms import (
    simulated_annealing_continuous as simulated_annealing,
    simulated_annealing_graph,
)
from .evolution_based_algorithms import genetic_algorithm, differential_evolution
from .biology_based_algorithms import (
    ant_colony_optimization, particle_swarm_optimization,
    artificial_bee_colony, firefly_algorithm, cuckoo_search,
)
from .human_based_algorithm import teaching_learning_based_optimization
from .continuous_classic_algorithms import (
    bfs_continuous, dfs_continuous, greedy_continuous,
    astar_continuous, ucs_continuous, hc_continuous, sa_continuous,
)

# ── Metaheuristics (same interface: objective, bounds, pop_size, max_iter) ────
META_ALGORITHMS = {
    "GA":   genetic_algorithm,
    "DE":   differential_evolution,
    "SA":   simulated_annealing,
    "ACO":  ant_colony_optimization,
    "PSO":  particle_swarm_optimization,
    "ABC":  artificial_bee_colony,
    "FA":   firefly_algorithm,
    "CS":   cuckoo_search,
    "TLBO": teaching_learning_based_optimization,
}

# ── Classic graph/combinatorial algorithms (graph_cost, heuristic, start, goal) ─
CLASSIC_ALGOS = {
    "BFS":    bfs,
    "DFS":    dfs,
    "UCS":    ucs,
    "Greedy": greedy_best_first_search,
    "A*":     a_star_search,
    "HC":     hill_climbing_steepest_ascent,
    "SA":     simulated_annealing_graph,
}

# ── Classic algorithms for CONTINUOUS functions (same interface as META_ALGORITHMS) ─
CONTINUOUS_CLASSIC_ALGOS = {
    "BFS":    bfs_continuous,
    "DFS":    dfs_continuous,
    "Greedy": greedy_continuous,
    "A*":     astar_continuous,
    "UCS":    ucs_continuous,
    "HC":     hc_continuous,
    "SA":     sa_continuous,
}
