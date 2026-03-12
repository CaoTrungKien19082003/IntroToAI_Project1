"""
=============================================================================
  Combinatorial Problem Solver  —  Strict NumPy Edition
  Solves TSP, Knapsack (KP), and Graph Coloring (GCP)
  using BFS, DFS, Greedy, and A*

  STRICTLY NumPy only — no heapq, no deque, no set, no itertools.
  Every data structure is a pre-allocated numpy ndarray.

  Replacement map
  ────────────────────────────────────────────────────────────────
  heapq      → NpMinHeap  : float64 array [CAP, 1+row_w]
                             col-0 = priority; pop = np.argmin(col-0)
  deque      → NpQueue    : circular float64 array [CAP, row_w]
                             head / tail int64 index scalars
  list stack → NpStack    : float64 array [CAP, row_w]
                             top int64 index scalar
  set        → numpy ops  : np.isin / np.unique / boolean masking

  State encoding (all packed as float64 rows)
  ────────────────────────────────────────────
  TSP  BFS/DFS row : [node, cost, visited[n],   path[n+1]]      2n+3 cols
  TSP  A*   row    : [f, g, node, visited[n],   path[n+1]]      2n+4 cols
  KP   BFS/DFS row : [idx, cap, val, mask[n]]                    n+3 cols
  KP   A*   row    : [neg_ub, neg_val, idx, cap, mask[n]]        n+4 cols
  GCP  BFS/DFS row : [vi, assign[n]]                             n+1 cols
  GCP  A*   row    : [g, vi, assign[n]]                          n+2 cols

  Bool arrays (visited mask, KP selection mask) stored as 0.0 / 1.0.
  Unused path slots padded with -1.0.
=============================================================================
"""

import time
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  NUMPY DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class NpQueue:
    """
    FIFO queue — circular float64 array [CAP, row_w].
    enqueue : buf[tail % CAP] = row  ;  tail += 1
    dequeue : row = buf[head % CAP]  ;  head += 1
    """
    def __init__(self, capacity, row_w):
        self._buf  = np.empty((capacity, row_w), dtype=np.float64)
        self._head = np.zeros(1, dtype=np.int64)
        self._tail = np.zeros(1, dtype=np.int64)
        self._cap  = np.int64(capacity)

    def empty(self):
        return bool(self._head[0] == self._tail[0])

    def enqueue(self, row):
        self._buf[int(self._tail[0] % self._cap)] = row
        self._tail[0] += np.int64(1)

    def dequeue(self):
        row = self._buf[int(self._head[0] % self._cap)].copy()
        self._head[0] += np.int64(1)
        return row


class NpStack:
    """
    LIFO stack — float64 array [CAP, row_w].
    push : buf[top] = row  ;  top += 1
    pop  : top -= 1  ;  row = buf[top]
    """
    def __init__(self, capacity, row_w):
        self._buf = np.empty((capacity, row_w), dtype=np.float64)
        self._top = np.zeros(1, dtype=np.int64)

    def empty(self):
        return bool(self._top[0] == np.int64(0))

    def push(self, row):
        self._buf[int(self._top[0])] = row
        self._top[0] += np.int64(1)

    def pop(self):
        self._top[0] -= np.int64(1)
        return self._buf[int(self._top[0])].copy()


class NpMinHeap:
    """
    Min-heap — float64 array [CAP, 1+row_w].  col-0 = priority.
    push : buf[size] = [priority, *row]  ;  size += 1
    pop  : idx = np.argmin(buf[:size, 0])
           swap buf[idx] ↔ buf[size-1]  ;  size -= 1
    """
    def __init__(self, capacity, row_w):
        self._buf  = np.empty((capacity, 1 + row_w), dtype=np.float64)
        self._size = np.zeros(1, dtype=np.int64)

    def empty(self):
        return bool(self._size[0] == np.int64(0))

    def push(self, priority, row):
        n = int(self._size[0])
        self._buf[n, 0]  = priority
        self._buf[n, 1:] = row
        self._size[0] += np.int64(1)

    def pop(self):
        n   = int(self._size[0])
        idx = int(np.argmin(self._buf[:n, 0]))
        row = self._buf[idx].copy()
        self._buf[idx] = self._buf[n - 1]
        self._size[0] -= np.int64(1)
        return row[0], row[1:]       # priority, payload


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
    bar = "═" * 62
    print(f"\n╔{bar}╗\n║  {title:<60}║\n╚{bar}╝")

def subsection(title):
    print(f"\n  ┌── {title}")


# ─────────────────────────────────────────────────────────────────────────────
#  TSP  —  Travelling Salesman Problem
#
#  State row (BFS/DFS): [node, cost, visited[n], path[n+1]]   width = 2n+3
#  State row (A*)     : [f, g, node, visited[n], path[n+1]]   width = 2n+4
#
#  visited stored as float64 0.0/1.0; path slots unused → -1.0
# ─────────────────────────────────────────────────────────────────────────────

class TSP:
    def __init__(self, dist):
        self.dist = np.asarray(dist, dtype=np.float64)
        self.n    = self.dist.shape[0]
        self._RW  = 2 * self.n + 3          # BFS/DFS row width
        self._RWA = 2 * self.n + 4          # A* row width
        self._CAP = max(5000, 2 ** self.n * self.n)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _pack(self, node, cost, visited, path, path_len):
        """Pack state into a fixed-width float64 row."""
        row = np.full(self._RW, -1.0, dtype=np.float64)
        row[0] = np.float64(node)
        row[1] = np.float64(cost)
        row[2:2+self.n] = visited.astype(np.float64)
        row[2+self.n:2+self.n+path_len] = path[:path_len].astype(np.float64)
        return row

    def _unpack(self, row):
        node     = int(row[0])
        cost     = float(row[1])
        visited  = row[2:2+self.n] > 0.5                # float → bool
        raw_path = row[2+self.n:]
        path_len = int(np.sum(raw_path >= 0))            # count non-(-1) slots
        path     = raw_path[:path_len].astype(np.int64)
        return node, cost, visited, path, path_len

    def _pack_astar(self, f, g, node, visited, path, path_len):
        row = np.full(self._RWA, -1.0, dtype=np.float64)
        row[0] = np.float64(f)
        row[1] = np.float64(g)
        row[2] = np.float64(node)
        row[3:3+self.n] = visited.astype(np.float64)
        row[3+self.n:3+self.n+path_len] = path[:path_len].astype(np.float64)
        return row

    def _unpack_astar(self, row):
        f        = float(row[0])
        g        = float(row[1])
        node     = int(row[2])
        visited  = row[3:3+self.n] > 0.5
        raw_path = row[3+self.n:]
        path_len = int(np.sum(raw_path >= 0))
        path     = raw_path[:path_len].astype(np.int64)
        return f, g, node, visited, path, path_len

    def _h_min_out(self, unvisited_mask):
        """Admissible heuristic: min outgoing edge per unvisited city."""
        idx = np.where(unvisited_mask)[0]
        if idx.size == 0:
            return np.float64(0.0)
        rows = self.dist[idx].copy()
        rows[np.arange(idx.size), idx] = np.inf
        return np.float64(np.min(rows, axis=1).sum())

    # ── BFS ───────────────────────────────────────────────────────────────────

    @timer
    def bfs(self):
        n = self.n
        best_cost = np.inf
        best_path = np.array([], dtype=np.int64)

        vis0  = np.zeros(n, dtype=bool); vis0[0] = True
        path0 = np.zeros(n+1, dtype=np.int64); path0[0] = 0
        q     = NpQueue(self._CAP, self._RW)
        q.enqueue(self._pack(0, 0.0, vis0, path0, 1))

        while not q.empty():
            node, cost, visited, path, plen = self._unpack(q.dequeue())
            if cost >= best_cost:
                continue
            if visited.all():
                total = cost + self.dist[node, 0]
                if total < best_cost:
                    best_cost = float(total)
                    best_path = np.append(path, np.int64(0))
                continue
            unvisited_idx = np.where(~visited)[0]
            for nxt in unvisited_idx:
                new_cost = cost + float(self.dist[node, nxt])
                if new_cost < best_cost:
                    new_v = visited.copy(); new_v[nxt] = True
                    new_p = path.copy()
                    new_p_full = np.full(n+1, -1, dtype=np.int64)
                    new_p_full[:plen] = new_p
                    new_p_full[plen]  = np.int64(nxt)
                    q.enqueue(self._pack(nxt, new_cost, new_v,
                                         new_p_full, plen+1))

        return best_path, best_cost

    # ── DFS ───────────────────────────────────────────────────────────────────

    @timer
    def dfs(self):
        n = self.n
        best_cost = np.inf
        best_path = np.array([], dtype=np.int64)

        vis0  = np.zeros(n, dtype=bool); vis0[0] = True
        path0 = np.zeros(n+1, dtype=np.int64); path0[0] = 0
        stk   = NpStack(self._CAP, self._RW)
        stk.push(self._pack(0, 0.0, vis0, path0, 1))

        while not stk.empty():
            node, cost, visited, path, plen = self._unpack(stk.pop())
            if cost >= best_cost:
                continue
            if visited.all():
                total = cost + self.dist[node, 0]
                if total < best_cost:
                    best_cost = float(total)
                    best_path = np.append(path, np.int64(0))
                continue
            # push cheapest neighbours last → popped first
            unvisited_idx = np.where(~visited)[0]
            costs_nxt = self.dist[node, unvisited_idx]
            order     = np.argsort(costs_nxt)[::-1]          # worst first
            for i in order:
                nxt      = int(unvisited_idx[i])
                new_cost = cost + float(self.dist[node, nxt])
                if new_cost < best_cost:
                    new_v = visited.copy(); new_v[nxt] = True
                    new_p_full = np.full(n+1, -1, dtype=np.int64)
                    new_p_full[:plen] = path
                    new_p_full[plen]  = np.int64(nxt)
                    stk.push(self._pack(nxt, new_cost, new_v,
                                        new_p_full, plen+1))

        return best_path, best_cost

    # ── Greedy (Nearest Neighbour, all starts) ────────────────────────────────

    @timer
    def greedy(self):
        n = self.n
        best_cost = np.inf
        best_path = np.array([], dtype=np.int64)

        # Try every start city using a pre-allocated path buffer
        path_buf = np.zeros(n+1, dtype=np.int64)

        for start in np.arange(n, dtype=np.int64):
            visited = np.zeros(n, dtype=bool); visited[start] = True
            path_buf[0] = start
            current = int(start)
            cost    = np.float64(0.0)

            for step in range(1, n):
                row = self.dist[current].copy()
                row[visited] = np.inf                    # mask visited with inf
                nxt = int(np.argmin(row))
                cost += self.dist[current, nxt]
                visited[nxt] = True
                path_buf[step] = np.int64(nxt)
                current = nxt

            cost += self.dist[current, int(start)]
            path_buf[n] = start

            if cost < best_cost:
                best_cost = float(cost)
                best_path = path_buf.copy()

        return best_path, best_cost

    # ── A* ────────────────────────────────────────────────────────────────────

    @timer
    def astar(self):
        n = self.n
        best_cost = np.inf
        best_path = np.array([], dtype=np.int64)

        vis0  = np.zeros(n, dtype=bool); vis0[0] = True
        path0 = np.full(n+1, -1, dtype=np.int64); path0[0] = 0
        h0    = self._h_min_out(~vis0)
        heap  = NpMinHeap(self._CAP, self._RWA)
        heap.push(float(h0), self._pack_astar(h0, 0.0, 0, vis0, path0, 1))

        while not heap.empty():
            priority, row = heap.pop()
            f, g, node, visited, path, plen = self._unpack_astar(row)
            if f >= best_cost:
                continue
            if visited.all():
                total = g + float(self.dist[node, 0])
                if total < best_cost:
                    best_cost = total
                    best_path = np.append(path, np.int64(0))
                continue
            unvisited_idx = np.where(~visited)[0]
            for nxt in unvisited_idx:
                new_g  = g + float(self.dist[node, nxt])
                new_v  = visited.copy(); new_v[nxt] = True
                h      = float(self._h_min_out(~new_v))
                # cheapest return-to-start edge from any non-start visited node
                ns     = np.where(new_v)[0]; ns = ns[ns != 0]
                if ns.size > 0:
                    h += float(np.min(self.dist[ns, 0]))
                new_f  = new_g + h
                if new_f < best_cost:
                    new_p = np.full(n+1, -1, dtype=np.int64)
                    new_p[:plen] = path
                    new_p[plen]  = np.int64(nxt)
                    heap.push(new_f, self._pack_astar(new_f, new_g, nxt,
                                                      new_v, new_p, plen+1))

        return best_path, best_cost


# ─────────────────────────────────────────────────────────────────────────────
#  KP  —  0/1 Knapsack Problem
#
#  State row (BFS/DFS): [idx, cap, val, mask[n]]     n+3 cols
#  State row (A*)     : [neg_ub, neg_val, idx, cap, mask[n]]  n+4 cols
#
#  mask stored as float64 0.0 / 1.0
# ─────────────────────────────────────────────────────────────────────────────

class KP:
    def __init__(self, weights, values, capacity):
        self.w    = np.asarray(weights,  dtype=np.float64)
        self.v    = np.asarray(values,   dtype=np.float64)
        self.cap  = np.float64(capacity)
        self.n    = len(weights)
        # Pre-sort indices by decreasing value/weight ratio
        self._ord = np.argsort(self.v / self.w)[::-1]
        self._CAP = max(1000, 2 ** self.n + 10)
        self._RW  = self.n + 3          # BFS/DFS row width
        self._RWA = self.n + 4          # A* row width

    # ── helpers ───────────────────────────────────────────────────────────────

    def _pack(self, idx, cap, val, mask):
        row = np.empty(self._RW, dtype=np.float64)
        row[0] = np.float64(idx)
        row[1] = np.float64(cap)
        row[2] = np.float64(val)
        row[3:] = mask.astype(np.float64)
        return row

    def _unpack(self, row):
        return (int(row[0]), float(row[1]),
                float(row[2]), row[3:] > 0.5)

    def _pack_astar(self, neg_ub, neg_val, idx, cap, mask):
        row = np.empty(self._RWA, dtype=np.float64)
        row[0] = np.float64(neg_ub)
        row[1] = np.float64(neg_val)
        row[2] = np.float64(idx)
        row[3] = np.float64(cap)
        row[4:] = mask.astype(np.float64)
        return row

    def _unpack_astar(self, row):
        return (float(row[0]), float(row[1]),
                int(row[2]), float(row[3]), row[4:] > 0.5)

    def _upper_bound(self, from_idx, remaining, val):
        """
        Fractional-knapsack upper bound via numpy cumsum.
        Only considers items with original index >= from_idx.
        """
        idx = self._ord[self._ord >= np.int64(from_idx)]
        if idx.size == 0:
            return np.float64(val)
        ws  = self.w[idx]; vs = self.v[idx]
        cum = np.cumsum(ws)
        fits = cum <= remaining
        val += float(np.sum(vs[fits]))
        rem  = remaining - float(np.sum(ws[fits]))
        pi   = int(np.sum(fits))
        if pi < vs.size and rem > 0:
            val += float(vs[pi]) * (rem / float(ws[pi]))
        return np.float64(val)

    # ── BFS ───────────────────────────────────────────────────────────────────

    @timer
    def bfs(self):
        best_val  = np.float64(0.0)
        best_mask = np.zeros(self.n, dtype=bool)
        mask0     = np.zeros(self.n, dtype=bool)
        q         = NpQueue(self._CAP, self._RW)
        q.enqueue(self._pack(0, float(self.cap), 0.0, mask0))

        while not q.empty():
            idx, cap, val, mask = self._unpack(q.dequeue())
            if val > best_val:
                best_val  = np.float64(val)
                best_mask = mask.copy()
            if idx == self.n:
                continue
            if self._upper_bound(idx, cap, val) <= best_val:
                continue
            # Don't take item idx
            q.enqueue(self._pack(idx+1, cap, val, mask))
            # Take item idx
            if self.w[idx] <= cap:
                new_mask      = mask.copy()
                new_mask[idx] = True
                q.enqueue(self._pack(idx+1,
                                     cap - float(self.w[idx]),
                                     val + float(self.v[idx]),
                                     new_mask))

        return np.where(best_mask)[0].tolist(), int(best_val)

    # ── DFS ───────────────────────────────────────────────────────────────────

    @timer
    def dfs(self):
        best_val  = np.float64(0.0)
        best_mask = np.zeros(self.n, dtype=bool)
        mask0     = np.zeros(self.n, dtype=bool)
        stk       = NpStack(self._CAP, self._RW)
        stk.push(self._pack(0, float(self.cap), 0.0, mask0))

        while not stk.empty():
            idx, cap, val, mask = self._unpack(stk.pop())
            if val > best_val:
                best_val  = np.float64(val)
                best_mask = mask.copy()
            if idx == self.n:
                continue
            if self._upper_bound(idx, cap, val) <= best_val:
                continue
            # Push don't-take first → take is explored first (LIFO)
            stk.push(self._pack(idx+1, cap, val, mask))
            if self.w[idx] <= cap:
                new_mask      = mask.copy()
                new_mask[idx] = True
                stk.push(self._pack(idx+1,
                                    cap - float(self.w[idx]),
                                    val + float(self.v[idx]),
                                    new_mask))

        return np.where(best_mask)[0].tolist(), int(best_val)

    # ── Greedy (by value/weight ratio) ────────────────────────────────────────

    @timer
    def greedy(self):
        """Take items in decreasing v/w order while capacity allows."""
        ws   = self.w[self._ord]
        vs   = self.v[self._ord]
        cum  = np.cumsum(ws)
        fits = cum <= self.cap                       # numpy boolean mask
        selected = self._ord[fits]
        total_v  = float(np.sum(vs[fits]))
        return np.sort(selected).tolist(), int(total_v)

    # ── A* ────────────────────────────────────────────────────────────────────

    @timer
    def astar(self):
        best_val  = np.float64(0.0)
        best_mask = np.zeros(self.n, dtype=bool)
        mask0     = np.zeros(self.n, dtype=bool)
        ub0       = self._upper_bound(0, float(self.cap), 0.0)
        heap      = NpMinHeap(self._CAP, self._RWA)
        heap.push(float(-ub0),
                  self._pack_astar(-ub0, 0.0, 0, float(self.cap), mask0))

        while not heap.empty():
            _, row    = heap.pop()
            neg_ub, neg_val, idx, cap, mask = self._unpack_astar(row)
            val = -neg_val
            if val > best_val:
                best_val  = np.float64(val)
                best_mask = mask.copy()
            if idx == self.n:
                continue
            if self._upper_bound(idx, cap, val) <= best_val:
                continue
            # Don't take
            ub_skip = self._upper_bound(idx+1, cap, val)
            if ub_skip > best_val:
                heap.push(float(-ub_skip),
                          self._pack_astar(-ub_skip, -val,
                                           idx+1, cap, mask))
            # Take
            if self.w[idx] <= cap:
                nv = val + float(self.v[idx])
                nc = cap - float(self.w[idx])
                ub_take = self._upper_bound(idx+1, nc, nv)
                if ub_take > best_val:
                    new_mask = mask.copy(); new_mask[idx] = True
                    heap.push(float(-ub_take),
                              self._pack_astar(-ub_take, -nv,
                                               idx+1, nc, new_mask))

        return np.sort(np.where(best_mask)[0]).tolist(), int(best_val)


# ─────────────────────────────────────────────────────────────────────────────
#  GCP  —  Graph Coloring Problem
#
#  State row (BFS/DFS): [vi, assign[n]]     n+1 cols
#  State row (A*)     : [g, vi, assign[n]]  n+2 cols
#
#  assign stored as float64; uncolored = -1.0
#  Python set replaced by np.isin + np.arange everywhere
# ─────────────────────────────────────────────────────────────────────────────

class GCP:
    def __init__(self, num_vertices, edges):
        self.n    = num_vertices
        self.adj  = np.zeros((num_vertices, num_vertices), dtype=bool)
        for u, v in edges:
            self.adj[u, v] = self.adj[v, u] = True
        self.edges  = np.array(edges, dtype=np.int64)
        degrees     = self.adj.sum(axis=1)
        self._order = np.argsort(degrees)[::-1]
        self._CAP   = max(50000, num_vertices ** num_vertices)
        self._RW    = num_vertices + 1
        self._RWA   = num_vertices + 2

    # ── helpers ───────────────────────────────────────────────────────────────

    def _pack(self, vi, assign):
        row = np.empty(self._RW, dtype=np.float64)
        row[0]  = np.float64(vi)
        row[1:] = assign.astype(np.float64)
        return row

    def _unpack(self, row):
        return int(row[0]), row[1:].astype(np.int64)

    def _pack_astar(self, g, vi, assign):
        row = np.empty(self._RWA, dtype=np.float64)
        row[0]  = np.float64(g)
        row[1]  = np.float64(vi)
        row[2:] = assign.astype(np.float64)
        return row

    def _unpack_astar(self, row):
        return float(row[0]), int(row[1]), row[2:].astype(np.int64)

    def _neighbor_colors(self, assign, vertex):
        """Colors of neighbours of vertex (only colored ones)."""
        nb   = self.adj[vertex]
        cols = assign[nb]
        return cols[cols >= 0]

    def _available_colors(self, assign, vertex, best_k):
        """
        Return sorted numpy array of assignable colors for vertex.
        Replaces: palette - nb_colors  (python set difference).
        Uses np.isin for exclusion.
        """
        cur_max = int(np.max(assign)) + 1 if np.any(assign >= 0) else 0
        palette = np.arange(min(cur_max + 1, best_k - 1), dtype=np.int64)
        if palette.size == 0:
            return palette
        nb_cols = self._neighbor_colors(assign, vertex)
        return palette[~np.isin(palette, nb_cols)]         # numpy set-difference

    def _colors_used(self, assign):
        colored = assign[assign >= 0]
        return int(np.unique(colored).size) if colored.size > 0 else 0

    def _is_valid(self, assign):
        if self.edges.size == 0:
            return True
        u, v = self.edges[:, 0], self.edges[:, 1]
        return bool(np.all(assign[u] != assign[v]))

    # ── BFS ───────────────────────────────────────────────────────────────────

    @timer
    def bfs(self):
        best_k      = self.n + 1
        best_assign = np.full(self.n, -1, dtype=np.int64)
        init        = np.full(self.n, -1, dtype=np.int64)
        q           = NpQueue(self._CAP, self._RW)
        q.enqueue(self._pack(0, init))

        while not q.empty():
            vi, assign = self._unpack(q.dequeue())
            cur_max    = int(np.max(assign)) + 1 if np.any(assign >= 0) else 0
            if cur_max >= best_k:
                continue
            if vi == self.n:
                k = self._colors_used(assign)
                if k < best_k:
                    best_k = k; best_assign = assign.copy()
                continue
            vertex    = int(self._order[vi])
            available = self._available_colors(assign, vertex, best_k)
            for color in available:
                new_a         = assign.copy()
                new_a[vertex] = int(color)
                q.enqueue(self._pack(vi + 1, new_a))

        return best_assign, best_k

    # ── DFS ───────────────────────────────────────────────────────────────────

    @timer
    def dfs(self):
        best_k      = self.n + 1
        best_assign = np.full(self.n, -1, dtype=np.int64)
        init        = np.full(self.n, -1, dtype=np.int64)
        stk         = NpStack(self._CAP, self._RW)
        stk.push(self._pack(0, init))

        while not stk.empty():
            vi, assign = self._unpack(stk.pop())
            cur_max    = int(np.max(assign)) + 1 if np.any(assign >= 0) else 0
            if cur_max >= best_k:
                continue
            if vi == self.n:
                k = self._colors_used(assign)
                if k < best_k:
                    best_k = k; best_assign = assign.copy()
                continue
            vertex    = int(self._order[vi])
            available = self._available_colors(assign, vertex, best_k)
            # push in reverse order → smallest color popped first
            for color in available[::-1]:
                new_a         = assign.copy()
                new_a[vertex] = int(color)
                stk.push(self._pack(vi + 1, new_a))

        return best_assign, best_k

    # ── Greedy (Welsh-Powell) ─────────────────────────────────────────────────

    @timer
    def greedy(self):
        assign = np.full(self.n, -1, dtype=np.int64)
        all_colors = np.arange(self.n, dtype=np.int64)

        for vertex in self._order:
            nb_cols   = self._neighbor_colors(assign, vertex)
            # smallest color not in neighbour colors — via np.isin
            available = all_colors[~np.isin(all_colors, nb_cols)]
            assign[vertex] = int(available[0])

        return assign, self._colors_used(assign)

    # ── A* ────────────────────────────────────────────────────────────────────

    @timer
    def astar(self):
        best_k      = self.n + 1
        best_assign = np.full(self.n, -1, dtype=np.int64)
        init        = np.full(self.n, -1, dtype=np.int64)
        heap        = NpMinHeap(self._CAP, self._RWA)
        heap.push(0.0, self._pack_astar(0, 0, init))

        while not heap.empty():
            _, row       = heap.pop()
            g, vi, assign = self._unpack_astar(row)
            if g >= best_k:
                continue
            if vi == self.n:
                k = self._colors_used(assign)
                if k < best_k:
                    best_k = k; best_assign = assign.copy()
                continue
            vertex    = int(self._order[vi])
            available = self._available_colors(assign, vertex, best_k)
            for color in available:
                new_a         = assign.copy()
                new_a[vertex] = int(color)
                new_g = float(np.unique(new_a[new_a >= 0]).size)
                if new_g < best_k:
                    heap.push(new_g,
                              self._pack_astar(new_g, vi + 1, new_a))

        return best_assign, best_k


# ─────────────────────────────────────────────────────────────────────────────
#  DEMO
# ─────────────────────────────────────────────────────────────────────────────

def run_tsp():
    section("TSP – Travelling Salesman Problem")
    dist = np.array([
        [ 0, 10, 15, 20, 25],
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


def run_kp():
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


def run_gcp():
    section("GCP – Graph Coloring Problem")
    n     = 6
    edges = [(0,1),(0,2),(1,2),(1,3),(2,4),(3,4),(3,5),(4,5),(0,5)]
    print(f"\n  Vertices : {n}")
    print(f"  Edges    : {edges}")
    gcp    = GCP(n, edges)
    colors = ["Red","Green","Blue","Yellow","Cyan","Magenta"]
    for name, fn in [("BFS", gcp.bfs), ("DFS", gcp.dfs),
                     ("Greedy", gcp.greedy), ("A*", gcp.astar)]:
        subsection(name)
        assignment, num_colors = fn()
        colored = {v: colors[c] for v, c in enumerate(assignment)}
        print(f"    Colors used : {num_colors}")
        print(f"    Assignment  : {colored}")
        print(f"    Valid       : {'✓ Yes' if gcp._is_valid(assignment) else '✗ No'}")


if __name__ == "__main__":
    print("\n" + "="*64)
    print("   Combinatorial Problem Solver  —  Strict NumPy Edition")
    print("   TSP · KP · GCP  ×  BFS · DFS · Greedy · A*")
    print(f"   NumPy {np.__version__}")
    print("="*64)
    run_tsp()
    run_kp()
    run_gcp()
    print("\n" + "="*64 + "\n  All demos complete.\n" + "="*64 + "\n")
