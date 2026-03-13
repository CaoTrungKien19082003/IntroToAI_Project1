import numpy as np
import random
import time

def teaching_learning_based_optimization(objective, bounds, pop_size, max_iter):
    n_dims = bounds.shape[0]
    learners = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, n_dims))
    fitness = np.array([objective(l) for l in learners])
    for _ in range(max_iter):
        teacher_idx = np.argmin(fitness)
        teacher = learners[teacher_idx]
        mean = np.mean(learners, axis=0)
        for i in range(pop_size):
            tf = np.random.randint(1, 3)  # Teaching factor 1 or 2
            new_learner = learners[i] + np.random.rand(n_dims) * (teacher - tf * mean)
            new_learner = np.clip(new_learner, bounds[:, 0], bounds[:, 1])
            new_fit = objective(new_learner)
            if new_fit < fitness[i]:
                learners[i] = new_learner
                fitness[i] = new_fit
        for i in range(pop_size):
            j = np.random.choice(np.setdiff1d(np.arange(pop_size), [i]))
            if fitness[i] < fitness[j]:
                new_learner = learners[i] + np.random.rand(n_dims) * (learners[i] - learners[j])
            else:
                new_learner = learners[i] + np.random.rand(n_dims) * (learners[j] - learners[i])
            new_learner = np.clip(new_learner, bounds[:, 0], bounds[:, 1])
            new_fit = objective(new_learner)
            if new_fit < fitness[i]:
                learners[i] = new_learner
                fitness[i] = new_fit
    best_idx = np.argmin(fitness)
    return learners[best_idx], fitness[best_idx]
