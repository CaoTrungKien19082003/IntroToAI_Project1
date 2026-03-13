import numpy as np
import random
import time

def bfs(cost_matrix, heuristic, start, goal):
    """Breadth-First Search (BFS) - không dùng chi phí, heuristic"""
    n = cost_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    parent = np.full(n, -1, dtype=int)
    queue = [start]
    visited[start] = True
    
    while queue:
        current = queue.pop(0)
        if current == goal:
            break
        neighbors = np.where(cost_matrix[current] > 0)[0]
        for neighbor in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = current
                queue.append(neighbor)
    
    if parent[goal] == -1 and start != goal:
        return np.array([])
    
    path = []
    current = goal
    while current != -1:
        path.append(current)
        current = parent[current]
    return np.array(path[::-1])


def dfs(cost_matrix, heuristic, start, goal):
    """Depth-First Search (DFS)"""
    n = cost_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    parent = np.full(n, -1, dtype=int)
    stack = [start]
    
    while stack:
        current = stack.pop()
        if not visited[current]:
            visited[current] = True
            if current == goal:
                break
            neighbors = np.where(cost_matrix[current] > 0)[0]
            for neighbor in neighbors[::-1]:
                if not visited[neighbor]:
                    parent[neighbor] = current
                    stack.append(neighbor)
    
    if parent[goal] == -1 and start != goal:
        return np.array([])
    
    path = []
    current = goal
    while current != -1:
        path.append(current)
        current = parent[current]
    return np.array(path[::-1])


def ucs(cost_matrix, heuristic, start, goal):
    """Uniform Cost Search (Dijkstra)"""
    n = cost_matrix.shape[0]
    dist = np.full(n, np.inf)
    dist[start] = 0
    parent = np.full(n, -1, dtype=int)
    pq = [(0, start)]  # (cost, node)
    visited = np.zeros(n, dtype=bool)
    
    while pq:
        pq.sort(key=lambda x: x[0])
        cost, current = pq.pop(0)
        if visited[current]:
            continue
        visited[current] = True
        if current == goal:
            break
        neighbors = np.where(cost_matrix[current] > 0)[0]
        for neighbor in neighbors:
            new_cost = cost + cost_matrix[current, neighbor]
            if new_cost < dist[neighbor]:
                dist[neighbor] = new_cost
                parent[neighbor] = current
                pq.append((new_cost, neighbor))
    
    if dist[goal] == np.inf:
        return np.array([])
    
    path = []
    current = goal
    while current != -1:
        path.append(current)
        current = parent[current]
    return np.array(path[::-1])


def greedy_best_first_search(cost_matrix, heuristic, start, goal):
    """Greedy Best-First Search"""
    n = cost_matrix.shape[0]
    parent = np.full(n, -1, dtype=int)
    pq = [(heuristic[start], start)]
    visited = np.zeros(n, dtype=bool)
    
    while pq:
        pq.sort(key=lambda x: x[0])
        _, current = pq.pop(0)
        if visited[current]:
            continue
        visited[current] = True
        if current == goal:
            break
        neighbors = np.where(cost_matrix[current] > 0)[0]
        for neighbor in neighbors:
            if not visited[neighbor]:
                parent[neighbor] = current
                pq.append((heuristic[neighbor], neighbor))
    
    if parent[goal] == -1 and start != goal:
        return np.array([])
    
    path = []
    current = goal
    while current != -1:
        path.append(current)
        current = parent[current]
    return np.array(path[::-1])


def a_star_search(cost_matrix, heuristic, start, goal):
    """A* Search"""
    n = cost_matrix.shape[0]
    g = np.full(n, np.inf)
    g[start] = 0
    parent = np.full(n, -1, dtype=int)
    pq = [(heuristic[start], start)]
    visited = np.zeros(n, dtype=bool)
    
    while pq:
        pq.sort(key=lambda x: x[0])
        f, current = pq.pop(0)
        if visited[current]:
            continue
        visited[current] = True
        if current == goal:
            break
        neighbors = np.where(cost_matrix[current] > 0)[0]
        for neighbor in neighbors:
            new_g = g[current] + cost_matrix[current, neighbor]
            if new_g < g[neighbor]:
                g[neighbor] = new_g
                f_new = new_g + heuristic[neighbor]
                parent[neighbor] = current
                pq.append((f_new, neighbor))
    
    if g[goal] == np.inf:
        return np.array([])
    
    path = []
    current = goal
    while current != -1:
        path.append(current)
        current = parent[current]
    return np.array(path[::-1])


def hill_climbing_steepest_ascent(cost_matrix, heuristic, start, goal):
    """Hill Climbing (Steepest Ascent) - minimization of heuristic"""
    current = start
    path = [current]
    while True:
        if current == goal:
            break
        neighbors = np.where(cost_matrix[current] > 0)[0]
        if len(neighbors) == 0:
            break
        neighbor_heurs = heuristic[neighbors]
        best_idx = np.argmin(neighbor_heurs)
        best_neighbor = neighbors[best_idx]
        if heuristic[best_neighbor] >= heuristic[current]:
            break
        current = best_neighbor
        path.append(current)
    
    if path[-1] != goal:
        return np.array([])
    return np.array(path)


