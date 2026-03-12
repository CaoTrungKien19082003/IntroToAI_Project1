"""
=============================================================================
  Combinatorial Problem Solver  —  NumPy Edition
  Solves TSP, Knapsack (KP), and Graph Coloring (GCP)
  using BFS, DFS, Greedy, and A*

  All state vectors, cost calculations, heuristics, and data structures
  are expressed exclusively through NumPy arrays and operations.
=============================================================================
"""

import heapq
import time
from collections import deque

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def timer(fn):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        print(f"    ⏱  {(time.perf_counter()-t0)*1000:.3f} ms")
        return result
    return wrapper

def section(title):
    bar = "═" * 60
    print(f"\n╔{bar}╗\n║  {title:<58}║\n╚{bar}╝")

def subsection(title):
    print(f"\n  ┌── {title}")


# ─────────────────────────────────────────────────────────────────────────────
#  TSP  —  Travelling Salesman Problem
#
#  State: np.ndarray bool[n]  — visited mask
#  Cost : scalar computed via dist[row, col] indexing
# ─────────────────────────────────────────────────────────────────────────────

class TSP:
    """
    State vector: 1-D boolean numpy array of length n.
      visited[i] == True  →  city i has been included in the current path.

    Edge lookups : dist[u, v]  — a single numpy scalar fetch.
    Heuristic    : np.min(dist[unvisited], axis=1).sum()  (A*).
    """

    def __init__(self, dist: np.ndarray):
        self.dist = np.asarray(dist, dtype=np.float64)
        self.n = self.dist.shape[0]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _route_cost(self, path: np.ndarray) -> float:
        # Vectorised: sum dist[path[i], path[i+1]] for all i
        return float(np.sum(self.dist[path[:-1], path[1:]]))

    def _h_min_out(self, unvisited_mask: np.ndarray) -> float:
        """
        Admissible heuristic: for every unvisited city, take the minimum
        outgoing edge to ANY other city (excluding self-loops).
        Uses full distance matrix rows so single-city case works correctly.
        """
        idx = np.where(unvisited_mask)[0]
        if idx.size == 0:
            return 0.0
        # Extract rows for all unvisited cities — shape (k, n)
        # Fancy indexing returns a copy, safe to modify
        rows = self.dist[idx]                           # shape (k, n)
        rows[np.arange(len(idx)), idx] = np.inf         # mask self-loops
        return float(np.min(rows, axis=1).sum())

    # ── BFS ───────────────────────────────────────────────────────────────────

    @timer
    def bfs(self):
        best_cost = np.inf
        best_path = np.array([], dtype=int)

        visited0 = np.zeros(self.n, dtype=bool)
        visited0[0] = True
        # queue entry: (current_node int, visited ndarray, cost float, path ndarray)
        queue = deque([(0, visited0, 0.0, np.array([0]))])

        while queue:
            node, visited, cost, path = queue.popleft()
            if cost >= best_cost:
                continue
            if visited.all():
                total = cost + float(self.dist[node, 0])
                if total < best_cost:
                    best_cost = total
                    best_path = np.append(path, 0)
                continue
            unvisited_idx = np.where(~visited)[0]
            for nxt in unvisited_idx:
                new_cost = cost + float(self.dist[node, nxt])
                if new_cost < best_cost:
                    new_v = visited.copy(); new_v[nxt] = True
                    queue.append((int(nxt), new_v, new_cost,
                                  np.append(path, nxt)))

        return best_path, best_cost

    # ── DFS ───────────────────────────────────────────────────────────────────

    @timer
    def dfs(self):
        best_cost = np.inf
        best_path = np.array([], dtype=int)

        visited0 = np.zeros(self.n, dtype=bool)
        visited0[0] = True
        stack = [(0, visited0, 0.0, np.array([0]))]

        while stack:
            node, visited, cost, path = stack.pop()
            if cost >= best_cost:
                continue
            if visited.all():
                total = cost + float(self.dist[node, 0])
                if total < best_cost:
                    best_cost = total
                    best_path = np.append(path, 0)
                continue
            for nxt in np.where(~visited)[0]:
                new_cost = cost + float(self.dist[node, nxt])
                if new_cost < best_cost:
                    new_v = visited.copy(); new_v[nxt] = True
                    stack.append((int(nxt), new_v, new_cost,
                                  np.append(path, nxt)))

        return best_path, best_cost

    # ── Greedy (Nearest Neighbour) ─────────────────────────────────────────

    @timer
    def greedy(self):
        best_cost = np.inf
        best_path = np.array([], dtype=int)

        for start in range(self.n):
            visited = np.zeros(self.n, dtype=bool)
            visited[start] = True
            path = [start]
            current = start
            cost = 0.0

            for _ in range(self.n - 1):
                # Mask visited cities with inf, pick argmin
                row = self.dist[current].copy()
                row[visited] = np.inf
                nxt = int(np.argmin(row))
                cost += float(self.dist[current, nxt])
                visited[nxt] = True
                path.append(nxt)
                current = nxt

            cost += float(self.dist[current, start])
            path.append(start)

            if cost < best_cost:
                best_cost = cost
                best_path = np.array(path)

        return best_path, best_cost

    # ── A* ────────────────────────────────────────────────────────────────────

    @timer
    def astar(self):
        best_cost = np.inf
        best_path = np.array([], dtype=int)

        visited0 = np.zeros(self.n, dtype=bool)
        visited0[0] = True
        unvisited0 = ~visited0
        h0 = self._h_min_out(unvisited0)
        # heap entry: (f, g, node, visited bytes, path tuple)
        heap = [(h0, 0.0, 0, visited0.tobytes(), (0,))]

        while heap:
            f, g, node, vis_bytes, path_tup = heapq.heappop(heap)
            if f >= best_cost:
                continue
            visited = np.frombuffer(vis_bytes, dtype=bool).copy()
            if visited.all():
                total = g + float(self.dist[node, 0])
                if total < best_cost:
                    best_cost = total
                    best_path = np.array(path_tup + (0,))
                continue
            unvisited_idx = np.where(~visited)[0]
            for nxt in unvisited_idx:
                new_g = g + float(self.dist[node, nxt])
                new_v = visited.copy(); new_v[nxt] = True
                h = self._h_min_out(~new_v)
                # cheapest return to start from any non-start visited node
                non_start = np.where(new_v)[0]
                non_start = non_start[non_start != 0]
                if non_start.size > 0:
                    h += float(np.min(self.dist[non_start, 0]))
                new_f = new_g + h
                if new_f < best_cost:
                    heapq.heappush(heap,
                                   (new_f, new_g, int(nxt),
                                    new_v.tobytes(),
                                    path_tup + (int(nxt),)))

        return best_path, best_cost


# ─────────────────────────────────────────────────────────────────────────────
#  KP  —  0/1 Knapsack Problem
#
#  State: np.ndarray bool[n]  — selection mask
#  UB   : fractional-knapsack relaxation via numpy cumsum / clip
# ─────────────────────────────────────────────────────────────────────────────

class KP:
    """
    State vector: 1-D boolean numpy array — which items have been taken.
    Upper bound : vectorised fractional-knapsack using np.cumsum.
    Ratio sort  : np.argsort on value/weight.
    """

    def __init__(self, weights: np.ndarray, values: np.ndarray, capacity: int):
        self.w = np.asarray(weights, dtype=np.float64)
        self.v = np.asarray(values,  dtype=np.float64)
        self.cap = float(capacity)
        self.n = len(weights)
        # Pre-sort indices by decreasing value/weight ratio
        self._order = np.argsort(self.v / self.w)[::-1]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _upper_bound(self, from_idx: int, remaining: float, val: float) -> float:
        """
        Fractional-knapsack upper bound.
        Vectorised with numpy: cumulative weights, clip, fractional fill.
        """
        idx = self._order[self._order >= from_idx]
        if idx.size == 0:
            return val
        ws = self.w[idx]
        vs = self.v[idx]
        cum_w = np.cumsum(ws)
        fits = cum_w <= remaining                       # bool mask: items that fully fit
        val += float(np.sum(vs[fits]))
        remaining_after = remaining - float(np.sum(ws[fits]))
        # First item that doesn't fully fit → fractional
        partial_idx = int(np.sum(fits))
        if partial_idx < len(vs) and remaining_after > 0:
            val += vs[partial_idx] * (remaining_after / ws[partial_idx])
        return val

    # ── BFS ───────────────────────────────────────────────────────────────────

    @timer
    def bfs(self):
        best_val = 0.0
        best_mask = np.zeros(self.n, dtype=bool)

        # queue: (item_index, remaining_cap, value, selection_mask)
        queue = deque([(0, self.cap, 0.0, np.zeros(self.n, dtype=bool))])

        while queue:
            idx, cap, val, mask = queue.popleft()
            if val > best_val:
                best_val = val; best_mask = mask.copy()
            if idx == self.n:
                continue
            if self._upper_bound(idx, cap, val) <= best_val:
                continue
            # Don't take
            queue.append((idx+1, cap, val, mask))
            # Take
            if self.w[idx] <= cap:
                new_mask = mask.copy(); new_mask[idx] = True
                queue.append((idx+1, cap - self.w[idx],
                               val + self.v[idx], new_mask))

        return np.where(best_mask)[0].tolist(), int(best_val)

    # ── DFS ───────────────────────────────────────────────────────────────────

    @timer
    def dfs(self):
        best_val = 0.0
        best_mask = np.zeros(self.n, dtype=bool)

        stack = [(0, self.cap, 0.0, np.zeros(self.n, dtype=bool))]

        while stack:
            idx, cap, val, mask = stack.pop()
            if val > best_val:
                best_val = val; best_mask = mask.copy()
            if idx == self.n:
                continue
            if self._upper_bound(idx, cap, val) <= best_val:
                continue
            stack.append((idx+1, cap, val, mask))
            if self.w[idx] <= cap:
                new_mask = mask.copy(); new_mask[idx] = True
                stack.append((idx+1, cap - self.w[idx],
                               val + self.v[idx], new_mask))

        return np.where(best_mask)[0].tolist(), int(best_val)

    # ── Greedy ────────────────────────────────────────────────────────────────

    @timer
    def greedy(self):
        """
        Take items in decreasing value/weight order while capacity allows.
        Pure numpy: boolean mask built via cumsum comparison.
        """
        ws = self.w[self._order]
        vs = self.v[self._order]
        cum = np.cumsum(ws)
        fits = cum <= self.cap                          # bool mask over sorted order
        selected_sorted = self._order[fits]
        total_val = float(np.sum(vs[fits]))
        return np.sort(selected_sorted).tolist(), int(total_val)

    # ── A* ────────────────────────────────────────────────────────────────────

    @timer
    def astar(self):
        best_val = 0.0
        best_mask = np.zeros(self.n, dtype=bool)

        ub0 = self._upper_bound(0, self.cap, 0.0)
        # heap: (-ub, -val, idx, remaining, mask_bytes)
        heap = [(-ub0, 0.0, 0, self.cap,
                 np.zeros(self.n, dtype=bool).tobytes())]

        while heap:
            neg_ub, neg_val, idx, cap, mask_bytes = heapq.heappop(heap)
            val = -neg_val
            if val > best_val:
                best_val = val
                best_mask = np.frombuffer(mask_bytes, dtype=bool).copy()
            if idx == self.n:
                continue
            if self._upper_bound(idx, cap, val) <= best_val:
                continue
            mask = np.frombuffer(mask_bytes, dtype=bool)
            # Don't take
            ub_skip = self._upper_bound(idx+1, cap, val)
            if ub_skip > best_val:
                heapq.heappush(heap, (-ub_skip, -val, idx+1, cap, mask_bytes))
            # Take
            if self.w[idx] <= cap:
                nv = val + self.v[idx]; nc = cap - self.w[idx]
                ub_take = self._upper_bound(idx+1, nc, nv)
                if ub_take > best_val:
                    new_mask = mask.copy(); new_mask[idx] = True
                    heapq.heappush(heap,
                                   (-ub_take, -nv, idx+1, nc,
                                    new_mask.tobytes()))

        return np.sort(np.where(best_mask)[0]).tolist(), int(best_val)


# ─────────────────────────────────────────────────────────────────────────────
#  GCP  —  Graph Coloring Problem
#
#  State: np.ndarray int[n]  — color assignment (-1 = uncolored)
#  Adj  : np.ndarray bool[n,n]  — adjacency matrix
#  Degree sort : np.argsort(row_sum(adj))[::-1]
# ─────────────────────────────────────────────────────────────────────────────

class GCP:
    """
    State vector: 1-D int numpy array, length n.
      assignment[v] == -1  → uncolored
      assignment[v] ==  c  → color c

    Adjacency matrix: boolean numpy array [n×n].
    Neighbour colors: adj_row & assignment via boolean masking.
    """

    def __init__(self, num_vertices: int, edges: list):
        self.n = num_vertices
        # Adjacency matrix
        self.adj = np.zeros((num_vertices, num_vertices), dtype=bool)
        for u, v in edges:
            self.adj[u, v] = self.adj[v, u] = True
        self.edges = np.array(edges, dtype=int)
        # Vertex ordering: most-constrained (highest degree) first
        degrees = self.adj.sum(axis=1)
        self._order = np.argsort(degrees)[::-1]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _neighbor_colors(self, assign: np.ndarray, vertex: int) -> np.ndarray:
        """Return array of colors used by neighbours of vertex."""
        nb_mask = self.adj[vertex]                      # bool row
        nb_colors = assign[nb_mask]                     # colors of neighbours
        return nb_colors[nb_colors >= 0]                # drop uncolored

    def _colors_used(self, assign: np.ndarray) -> int:
        colored = assign[assign >= 0]
        return int(np.unique(colored).size) if colored.size > 0 else 0

    def _is_valid(self, assign: np.ndarray) -> bool:
        """Vectorised validity check over all edges."""
        if self.edges.size == 0:
            return True
        u, v = self.edges[:, 0], self.edges[:, 1]
        return bool(np.all(assign[u] != assign[v]))

    # ── BFS ───────────────────────────────────────────────────────────────────

    @timer
    def bfs(self):
        best_k = self.n + 1
        best_assign = np.full(self.n, -1, dtype=int)

        init = np.full(self.n, -1, dtype=int)
        queue = deque([(0, init)])

        while queue:
            vi, assign = queue.popleft()
            cur_max = int(np.max(assign)) + 1 if np.any(assign >= 0) else 0
            if cur_max >= best_k:
                continue
            if vi == self.n:
                k = self._colors_used(assign)
                if k < best_k:
                    best_k = k; best_assign = assign.copy()
                continue
            vertex = int(self._order[vi])
            nb_colors = set(self._neighbor_colors(assign, vertex).tolist())
            palette = set(range(min(cur_max + 1, best_k - 1)))
            for color in sorted(palette - nb_colors):
                new_a = assign.copy(); new_a[vertex] = color
                queue.append((vi + 1, new_a))

        return best_assign, best_k

    # ── DFS ───────────────────────────────────────────────────────────────────

    @timer
    def dfs(self):
        best_k = self.n + 1
        best_assign = np.full(self.n, -1, dtype=int)

        init = np.full(self.n, -1, dtype=int)
        stack = [(0, init)]

        while stack:
            vi, assign = stack.pop()
            cur_max = int(np.max(assign)) + 1 if np.any(assign >= 0) else 0
            if cur_max >= best_k:
                continue
            if vi == self.n:
                k = self._colors_used(assign)
                if k < best_k:
                    best_k = k; best_assign = assign.copy()
                continue
            vertex = int(self._order[vi])
            nb_colors = set(self._neighbor_colors(assign, vertex).tolist())
            palette = set(range(min(cur_max + 1, best_k - 1)))
            for color in sorted(palette - nb_colors, reverse=True):
                new_a = assign.copy(); new_a[vertex] = color
                stack.append((vi + 1, new_a))

        return best_assign, best_k

    # ── Greedy (Welsh-Powell) ─────────────────────────────────────────────────

    @timer
    def greedy(self):
        """
        Assign each vertex the smallest color not used by any neighbour.
        Neighbour-color lookup via numpy boolean masking.
        """
        assign = np.full(self.n, -1, dtype=int)
        for vertex in self._order:
            nb_colors = set(self._neighbor_colors(assign, vertex).tolist())
            color = 0
            while color in nb_colors:
                color += 1
            assign[vertex] = color
        return assign, self._colors_used(assign)

    # ── A* ────────────────────────────────────────────────────────────────────

    @timer
    def astar(self):
        """
        Best-first search: priority = number of colors committed so far.
        State serialised as numpy array bytes for heap deduplication.
        """
        best_k = self.n + 1
        best_assign = np.full(self.n, -1, dtype=int)

        init = np.full(self.n, -1, dtype=int)
        # (g, vi, assign_bytes)
        heap = [(0, 0, init.tobytes())]

        while heap:
            g, vi, assign_bytes = heapq.heappop(heap)
            assign = np.frombuffer(assign_bytes, dtype=int).copy()
            if g >= best_k:
                continue
            if vi == self.n:
                k = self._colors_used(assign)
                if k < best_k:
                    best_k = k; best_assign = assign.copy()
                continue
            vertex = int(self._order[vi])
            nb_colors = set(self._neighbor_colors(assign, vertex).tolist())
            cur_max = int(np.max(assign)) + 1 if np.any(assign >= 0) else 0
            palette = set(range(min(cur_max + 1, best_k - 1)))
            for color in sorted(palette - nb_colors):
                new_a = assign.copy(); new_a[vertex] = color
                new_g = int(np.unique(new_a[new_a >= 0]).size)
                if new_g < best_k:
                    heapq.heappush(heap, (new_g, vi + 1, new_a.tobytes()))

        return best_assign, best_k


# ─────────────────────────────────────────────────────────────────────────────
#  DEMO / TEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_tsp_demo():
    section("TSP – Travelling Salesman Problem")
    dist = np.array([
        [0,  10, 15, 20, 25],
        [10,  0, 35, 25, 20],
        [15, 35,  0, 30, 10],
        [20, 25, 30,  0, 15],
        [25, 20, 10, 15,  0],
    ], dtype=np.float64)
    print("\n  Distance matrix (5 cities):\n", dist)
    tsp = TSP(dist)
    for name, fn in [("BFS", tsp.bfs), ("DFS", tsp.dfs),
                     ("Greedy", tsp.greedy), ("A*", tsp.astar)]:
        subsection(name)
        path, cost = fn()
        print(f"    Path : {' → '.join(map(str, path))}")
        print(f"    Cost : {cost:.1f}")


def run_kp_demo():
    section("KP – 0/1 Knapsack Problem")
    weights  = np.array([2, 3, 4, 5, 9, 4], dtype=np.float64)
    values   = np.array([3, 4, 5, 8, 10, 3], dtype=np.float64)
    capacity = 10
    print(f"\n  Items    : {np.arange(len(weights)).tolist()}")
    print(f"  Weights  : {weights.tolist()}")
    print(f"  Values   : {values.tolist()}")
    print(f"  Capacity : {capacity}")
    kp = KP(weights, values, capacity)
    for name, fn in [("BFS", kp.bfs), ("DFS", kp.dfs),
                     ("Greedy", kp.greedy), ("A*", kp.astar)]:
        subsection(name)
        items, val = fn()
        w = int(np.sum(weights[items]))
        print(f"    Items selected : {items}")
        print(f"    Total weight   : {w}  (capacity={capacity})")
        print(f"    Total value    : {val}")


def run_gcp_demo():
    section("GCP – Graph Coloring Problem")
    n = 6
    edges = [(0,1),(0,2),(1,2),(1,3),(2,4),(3,4),(3,5),(4,5),(0,5)]
    print(f"\n  Vertices : {n}")
    print(f"  Edges    : {edges}")
    gcp = GCP(n, edges)
    color_names = ["Red","Green","Blue","Yellow","Cyan","Magenta"]
    for name, fn in [("BFS", gcp.bfs), ("DFS", gcp.dfs),
                     ("Greedy", gcp.greedy), ("A*", gcp.astar)]:
        subsection(name)
        assignment, num_colors = fn()
        colored = {v: color_names[c] for v, c in enumerate(assignment)}
        print(f"    Colors used : {num_colors}")
        print(f"    Assignment  : {colored}")
        print(f"    Valid       : {'✓ Yes' if gcp._is_valid(assignment) else '✗ No'}")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*62)
    print("   Combinatorial Problem Solver  —  NumPy Edition")
    print("   TSP · KP · GCP  ×  BFS · DFS · Greedy · A*")
    print(f"   NumPy {np.__version__}")
    print("="*62)
    run_tsp_demo()
    run_kp_demo()
    run_gcp_demo()
    print("\n" + "="*62 + "\n  All demos complete.\n" + "="*62 + "\n")
