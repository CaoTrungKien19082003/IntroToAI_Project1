import numpy as np
import random
import time

def genetic_algorithm(objective, bounds, pop_size, max_iter):
    n_dims = bounds.shape[0]
    population = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, n_dims))
    fitness = np.array([objective(ind) for ind in population])
    
    for _ in range(max_iter):
        selected = []
        for __ in range(pop_size):
            idx = np.random.choice(pop_size, 2)
            selected.append(population[idx[np.argmin(fitness[idx])]])
        selected = np.array(selected)
        
        offspring = []
        for i in range(0, pop_size, 2):
            p1, p2 = selected[i], selected[(i+1)%pop_size]
            cp = np.random.randint(1, n_dims)
            c1 = np.concatenate((p1[:cp], p2[cp:]))
            c2 = np.concatenate((p2[:cp], p1[cp:]))
            offspring.extend([c1, c2])
        offspring = np.array(offspring[:pop_size])
        
        mutation_rate = 0.01
        for i in range(pop_size):
            if np.random.rand() < mutation_rate:
                offspring[i] += np.random.normal(0, 0.1, n_dims)
        
        offspring = np.clip(offspring, bounds[:, 0], bounds[:, 1])
        offspring_fitness = np.array([objective(ind) for ind in offspring])
        
        combined = np.vstack((population, offspring))
        combined_fitness = np.hstack((fitness, offspring_fitness))
        indices = np.argsort(combined_fitness)[:pop_size]
        population = combined[indices]
        fitness = combined_fitness[indices]
    
    best_idx = np.argmin(fitness)
    return population[best_idx], fitness[best_idx]


def differential_evolution(objective, bounds, pop_size, max_iter):
    n_dims = bounds.shape[0]
    population = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, n_dims))
    fitness = np.array([objective(ind) for ind in population])
    F = 0.8
    CR = 0.9
    
    for _ in range(max_iter):
        for i in range(pop_size):
            idx = np.random.choice(np.setdiff1d(np.arange(pop_size), [i]), 3, replace=False)
            mutant = population[idx[0]] + F * (population[idx[1]] - population[idx[2]])
            mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])
            
            cross_points = np.random.rand(n_dims) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, n_dims)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            trial_fitness = objective(trial)
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
    
    best_idx = np.argmin(fitness)
    return population[best_idx], fitness[best_idx]