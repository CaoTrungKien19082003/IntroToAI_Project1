import numpy as np
import random

def ant_colony_optimization(objective, bounds, pop_size=50, max_iter=200,
                             archive_size=50, q=0.1, xi=0.5):
    dim, lower, upper = bounds.shape[0], bounds[:, 0], bounds[:, 1]
    archive         = np.random.uniform(lower, upper, (archive_size, dim))
    archive_fitness = np.array([objective(s) for s in archive])
    best_solution, best_fitness = None, np.inf

    for _ in range(max_iter):
        solutions, fitness = [], []
        for __ in range(pop_size):
            probs = 1.0 / (q * archive_size * (1 + q * np.arange(archive_size)))
            probs = np.nan_to_num(np.clip(probs, 0, 1))
            s     = probs.sum()
            probs = probs / s if s > 0 else np.ones(archive_size) / archive_size
            idx   = np.random.choice(archive_size, p=probs)
            sigma = xi * np.std(archive, axis=0)
            sol   = np.clip(np.random.normal(archive[idx], sigma), lower, upper)
            fit   = objective(sol)
            solutions.append(sol); fitness.append(fit)
            if fit < best_fitness:
                best_fitness, best_solution = fit, sol.copy()
        combined         = np.vstack((archive, solutions))
        combined_fitness = np.nan_to_num(np.clip(
            np.concatenate((archive_fitness, fitness)), -1e10, 1e10),
            nan=np.inf, posinf=np.inf)
        elite            = np.argsort(combined_fitness)[:archive_size]
        archive, archive_fitness = combined[elite], combined_fitness[elite]
        if np.all(np.isinf(archive_fitness)):
            archive         = np.random.uniform(lower, upper, (archive_size, dim))
            archive_fitness = np.array([objective(s) for s in archive])

    return best_solution, best_fitness


def particle_swarm_optimization(objective, bounds, pop_size, max_iter):
    n   = bounds.shape[0]
    pos = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, n))
    vel = np.random.uniform(-1, 1, (pop_size, n))
    pb_pos = pos.copy()
    pb_fit = np.array([objective(p) for p in pos])
    gi     = np.argmin(pb_fit)
    gb_pos, gb_fit = pos[gi].copy(), pb_fit[gi]
    w, c1, c2 = 0.729, 1.494, 1.494
    for _ in range(max_iter):
        r1, r2 = np.random.rand(pop_size, n), np.random.rand(pop_size, n)
        vel    = w*vel + c1*r1*(pb_pos - pos) + c2*r2*(gb_pos - pos)
        pos    = np.clip(pos + vel, bounds[:, 0], bounds[:, 1])
        fit    = np.array([objective(p) for p in pos])
        imp    = fit < pb_fit
        pb_pos[imp], pb_fit[imp] = pos[imp], fit[imp]
        gi = np.argmin(pb_fit)
        gb_pos, gb_fit = pb_pos[gi].copy(), pb_fit[gi]
    return gb_pos, gb_fit


def artificial_bee_colony(objective, bounds, pop_size, max_iter):
    n   = bounds.shape[0]
    fs  = pop_size // 2
    pop = np.random.uniform(bounds[:, 0], bounds[:, 1], (fs, n))
    fit = np.array([objective(ind) for ind in pop])
    tri = np.zeros(fs)
    for _ in range(max_iter):
        for i in range(fs):
            k   = np.random.choice(np.setdiff1d(np.arange(fs), [i]))
            phi = np.random.uniform(-1, 1, n)
            nb  = np.clip(pop[i] + phi*(pop[i] - pop[k]), bounds[:, 0], bounds[:, 1])
            nf  = objective(nb)
            if nf < fit[i]:
                pop[i], fit[i], tri[i] = nb, nf, 0
            else:
                tri[i] += 1
        probs = np.maximum(1/(1+fit), 0)
        probs = (probs - probs.min() + 1e-10)
        probs /= probs.sum()
        for _ in range(fs):
            sel = np.random.choice(fs, p=probs)
            k   = np.random.choice(np.setdiff1d(np.arange(fs), [sel]))
            phi = np.random.uniform(-1, 1, n)
            nb  = np.clip(pop[sel] + phi*(pop[sel] - pop[k]), bounds[:, 0], bounds[:, 1])
            nf  = objective(nb)
            if nf < fit[sel]:
                pop[sel], fit[sel], tri[sel] = nb, nf, 0
            else:
                tri[sel] += 1
        lim = fs * n
        for i in range(fs):
            if tri[i] > lim:
                pop[i], fit[i], tri[i] = (
                    np.random.uniform(bounds[:, 0], bounds[:, 1], n),
                    objective(pop[i]), 0)
    bi = np.argmin(fit)
    return pop[bi], fit[bi]


def firefly_algorithm(objective, bounds, pop_size, max_iter):
    n = bounds.shape[0]
    lower, upper = bounds[:, 0], bounds[:, 1]
    pos = np.random.uniform(lower, upper, (pop_size, n))
    fit = np.array([objective(p) for p in pos])
    alpha, beta0, gamma = 0.2, 1.0, 1.0

    bi       = np.argmin(fit)
    best     = pos[bi].copy()
    best_fit = fit[bi]

    for _ in range(max_iter):
        diff = pos[:, None, :] - pos[None, :, :]
        sq_dist = (diff ** 2).sum(axis=-1)
        beta_mat = beta0 * np.exp(-gamma * sq_dist)

        attract = (fit[None, :] < fit[:, None]).astype(np.float64)
        weights = beta_mat * attract
        move = -(weights[:, :, None] * diff).sum(axis=1)
        move += alpha * np.random.uniform(-0.5, 0.5, (pop_size, n))

        new_pos = np.clip(pos + move, lower, upper)
        new_fit = np.array([objective(p) for p in new_pos])

        improved = new_fit < fit
        pos[improved] = new_pos[improved]
        fit[improved] = new_fit[improved]

        alpha *= 0.95

        bi = np.argmin(fit)
        if fit[bi] < best_fit:
            best_fit = fit[bi]
            best     = pos[bi].copy()

    return best, best_fit


def cuckoo_search(objective, bounds, pop_size, max_iter):
    n     = bounds.shape[0]
    nests = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, n))
    fit   = np.array([objective(k) for k in nests])
    pa    = 0.25
    for _ in range(max_iter):
        for i in range(pop_size):
            step = np.random.normal(0,1,n) / (np.random.rand(n)**(1/3))
            nb   = np.clip(nests[i] + 0.01*step*(nests[np.random.randint(pop_size)]-nests[i]),
                           bounds[:, 0], bounds[:, 1])
            j    = np.random.randint(pop_size)
            if objective(nb) < fit[j]:
                nests[j], fit[j] = nb, objective(nb)
        worst = int(pa * pop_size)
        for i in np.argsort(fit)[-worst:]:
            nests[i] = np.random.uniform(bounds[:, 0], bounds[:, 1], n)
            fit[i]   = objective(nests[i])
    bi = np.argmin(fit)
    return nests[bi], fit[bi]
