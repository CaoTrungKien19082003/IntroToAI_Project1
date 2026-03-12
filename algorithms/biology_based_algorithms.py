import numpy as np
import random
import time


def ant_colony_optimization(objective, bounds, pop_size=50, max_iter=200, archive_size=50, q=0.1, xi=0.5):
    """
    Ant Colony Optimization for continuous optimization (Gaussian kernel version)
    - objective: hàm fitness (càng nhỏ càng tốt)
    - bounds: array shape (dim, 2) [lower, upper] cho mỗi chiều
    - pop_size: số kiến mỗi thế hệ
    - max_iter: số thế hệ tối đa
    - archive_size: kích thước archive
    - q, xi: tham số Gaussian kernel
    """
    dim = bounds.shape[0]
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    # Khởi tạo archive ban đầu (ngẫu nhiên)
    archive = np.random.uniform(lower, upper, (archive_size, dim))
    archive_fitness = np.array([objective(sol) for sol in archive])

    best_solution = None
    best_fitness = np.inf

    for it in range(max_iter):
        solutions = []
        fitness = []

        for _ in range(pop_size):
            # Chọn giải pháp từ archive để làm mean
            probs = 1.0 / (q * archive_size * (1 + q * np.arange(archive_size)))
            probs /= probs.sum()

            # Xử lý probs nếu có NaN hoặc âm (fallback uniform)
            probs = np.nan_to_num(probs, nan=0.0)
            probs = np.clip(probs, 0.0, 1.0)
            probs_sum = probs.sum()
            if probs_sum <= 0:
                probs = np.ones(archive_size) / archive_size
            else:
                probs /= probs_sum

            idx = np.random.choice(archive_size, p=probs)
            mean = archive[idx]
            sigma = xi * np.std(archive, axis=0)  # độ lệch chuẩn toàn archive

            # Sinh giải pháp mới từ Gaussian kernel
            sol = np.random.normal(mean, sigma)
            sol = np.clip(sol, lower, upper)  # giới hạn trong bounds

            fit = objective(sol)
            solutions.append(sol)
            fitness.append(fit)

            if fit < best_fitness:
                best_fitness = fit
                best_solution = sol.copy()

        # Cập nhật archive: thêm mới, giữ elite
        combined = np.vstack((archive, solutions))
        combined_fitness = np.concatenate((archive_fitness, fitness))

        # Xử lý inf/NaN trong fitness trước khi chọn elite
        combined_fitness = np.nan_to_num(combined_fitness, nan=np.inf, posinf=np.inf, neginf=-np.inf)
        combined_fitness = np.clip(combined_fitness, -1e10, 1e10)

        # Lấy archive_size giải pháp tốt nhất (fitness nhỏ nhất)
        elite_indices = np.argsort(combined_fitness)[:archive_size]
        archive = combined[elite_indices]
        archive_fitness = combined_fitness[elite_indices]

        # Nếu toàn inf, fallback random
        if np.all(np.isinf(archive_fitness)):
            archive = np.random.uniform(lower, upper, (archive_size, dim))
            archive_fitness = np.array([objective(sol) for sol in archive])


    return best_solution, best_fitness

def particle_swarm_optimization(objective, bounds, pop_size, max_iter):
    """
    Particle Swarm Optimization (PSO) for minimization.
    :param objective: callable
    :param bounds: (n_dims, 2) np.array
    :param pop_size: int
    :param max_iter: int
    :return: (best_position np.array, best_fitness float)
    """
    n_dims = bounds.shape[0]
    positions = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, n_dims))
    velocities = np.random.uniform(-1, 1, (pop_size, n_dims))
    personal_best_pos = positions.copy()
    personal_best_fit = np.array([objective(p) for p in positions])
    global_best_idx = np.argmin(personal_best_fit)
    global_best_pos = positions[global_best_idx].copy()
    global_best_fit = personal_best_fit[global_best_idx]
    w = 0.729  # Inertia
    c1 = 1.494  # Cognitive
    c2 = 1.494  # Social
    for _ in range(max_iter):
        r1 = np.random.rand(pop_size, n_dims)
        r2 = np.random.rand(pop_size, n_dims)
        velocities = w * velocities + c1 * r1 * (personal_best_pos - positions) + c2 * r2 * (global_best_pos - positions)
        positions += velocities
        positions = np.clip(positions, bounds[:, 0], bounds[:, 1])
        fitness = np.array([objective(p) for p in positions])
        improved = fitness < personal_best_fit
        personal_best_pos[improved] = positions[improved]
        personal_best_fit[improved] = fitness[improved]
        global_best_idx = np.argmin(personal_best_fit)
        global_best_pos = personal_best_pos[global_best_idx].copy()
        global_best_fit = personal_best_fit[global_best_idx]
    return global_best_pos, global_best_fit

def artificial_bee_colony(objective, bounds, pop_size, max_iter):
    """
    Artificial Bee Colony (ABC) for minimization.
    :param objective: callable
    :param bounds: (n_dims, 2) np.array
    :param pop_size: int (number of food sources = pop_size // 2)
    :param max_iter: int
    :return: (best_position np.array, best_fitness float)
    """
    n_dims = bounds.shape[0]
    food_sources = pop_size // 2
    population = np.random.uniform(bounds[:, 0], bounds[:, 1], (food_sources, n_dims))
    fitness = np.array([objective(ind) for ind in population])
    trials = np.zeros(food_sources)
    for _ in range(max_iter):
        # Employed bees
        for i in range(food_sources):
            k = np.random.choice(np.setdiff1d(np.arange(food_sources), [i]))
            phi = np.random.uniform(-1, 1, n_dims)
            new_pos = population[i] + phi * (population[i] - population[k])
            new_pos = np.clip(new_pos, bounds[:, 0], bounds[:, 1])
            new_fit = objective(new_pos)
            if new_fit < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fit
                trials[i] = 0
            else:
                trials[i] += 1
        # Onlooker bees
        probs = 1 / (1 + fitness)
        probs = np.maximum(probs, 0)  # tránh âm
        probs_shifted = probs - np.min(probs) + 1e-10
        probs = probs_shifted / probs_shifted.sum()  # normalize an toàn
        for i in range(food_sources):
            selected = np.random.choice(food_sources, p=probs)
            k = np.random.choice(np.setdiff1d(np.arange(food_sources), [selected]))
            phi = np.random.uniform(-1, 1, n_dims)
            new_pos = population[selected] + phi * (population[selected] - population[k])
            new_pos = np.clip(new_pos, bounds[:, 0], bounds[:, 1])
            new_fit = objective(new_pos)
            if new_fit < fitness[selected]:
                population[selected] = new_pos
                fitness[selected] = new_fit
                trials[selected] = 0
            else:
                trials[selected] += 1
        # Scout bees
        limit = food_sources * n_dims
        for i in range(food_sources):
            if trials[i] > limit:
                population[i] = np.random.uniform(bounds[:, 0], bounds[:, 1], n_dims)
                fitness[i] = objective(population[i])
                trials[i] = 0
    best_idx = np.argmin(fitness)
    return population[best_idx], fitness[best_idx]

def firefly_algorithm(objective, bounds, pop_size, max_iter):
    """
    Firefly Algorithm (FA) for minimization.
    :param objective: callable
    :param bounds: (n_dims, 2) np.array
    :param pop_size: int
    :param max_iter: int
    :return: (best_position np.array, best_fitness float)
    """
    n_dims = bounds.shape[0]
    positions = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, n_dims))
    fitness = np.array([objective(p) for p in positions])
    alpha = 0.2  # Randomness
    beta0 = 1.0  # Attractiveness
    gamma = 1.0  # Absorption
    for _ in range(max_iter):
        for i in range(pop_size):
            for j in range(pop_size):
                if fitness[j] < fitness[i]:  # Brighter (lower fitness)
                    r = np.linalg.norm(positions[i] - positions[j])
                    beta = beta0 * np.exp(-gamma * r ** 2)
                    positions[i] += beta * (positions[j] - positions[i]) + alpha * np.random.uniform(-0.5, 0.5, n_dims)
                    positions[i] = np.clip(positions[i], bounds[:, 0], bounds[:, 1])
                    fitness[i] = objective(positions[i])
        alpha *= 0.95  # Decrease randomness
    best_idx = np.argmin(fitness)
    return positions[best_idx], fitness[best_idx]

def cuckoo_search(objective, bounds, pop_size, max_iter):
    """
    Cuckoo Search (CS) for minimization.
    :param objective: callable
    :param bounds: (n_dims, 2) np.array
    :param pop_size: int (number of nests)
    :param max_iter: int
    :return: (best_position np.array, best_fitness float)
    """
    n_dims = bounds.shape[0]
    nests = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, n_dims))
    fitness = np.array([objective(n) for n in nests])
    pa = 0.25  # Fraction of worse nests to abandon
    for _ in range(max_iter):
        # Generate new cuckoos by Levy flight
        for i in range(pop_size):
            step = np.random.normal(0, 1, n_dims) / (np.random.rand(n_dims) ** (1/3))  # Approximate Levy
            new_nest = nests[i] + 0.01 * step * (nests[np.random.randint(pop_size)] - nests[i])
            new_nest = np.clip(new_nest, bounds[:, 0], bounds[:, 1])
            j = np.random.randint(pop_size)
            if objective(new_nest) < fitness[j]:
                nests[j] = new_nest
                fitness[j] = objective(new_nest)
        # Abandon worse nests
        sorted_idx = np.argsort(fitness)
        worst = int(pa * pop_size)
        for i in range(worst):
            idx = sorted_idx[-i-1]
            nests[idx] = np.random.uniform(bounds[:, 0], bounds[:, 1], n_dims)
            fitness[idx] = objective(nests[idx])
    best_idx = np.argmin(fitness)
    return nests[best_idx], fitness[best_idx]