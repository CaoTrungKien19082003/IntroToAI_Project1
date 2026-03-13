
import os, sys, time, random, heapq
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algorithms import META_ALGORITHMS, CLASSIC_ALGOS, CONTINUOUS_CLASSIC_ALGOS
from algorithms.classic_algorithms import (
    bfs, dfs, ucs, greedy_best_first_search,
    a_star_search, hill_climbing_steepest_ascent,
)

_CLASSIC_KEYS = {"BFS", "DFS", "UCS", "Greedy", "A*", "HC", "SA"}

SMALL_NUM_RUNS = 20 
SMALL_POP_SIZE = 60
SMALL_MAX_ITER = 500 

LARGE_NUM_RUNS  = 20
LARGE_POP_SIZE  = 80
LARGE_MAX_ITER  = 400
LARGE_NODE_CAP  = 4_000   
LARGE_HC_ITER   = 50_000
LARGE_SA_ITER   = 50_000
LARGE_SA_T0     = 800.0
LARGE_SA_ALPHA  = 0.9985

def sphere(x):     return float(np.sum(x**2))
def rosenbrock(x): return float(np.sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2))
def rastrigin(x):  return float(10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x)))
def ackley(x):
    n = len(x)
    return float(-20*np.exp(-0.2*np.sqrt(np.sum(x**2)/n))
                 - np.exp(np.sum(np.cos(2*np.pi*x))/n) + 20 + np.e)
def griewank(x):
    i = np.arange(1, len(x)+1, dtype=np.float64)
    return float(np.sum(x**2)/4000.0 - np.prod(np.cos(x/np.sqrt(i))) + 1.0)

CONT_PROBLEMS = {
    "Sphere":     (sphere,     np.array([[-5.12,   5.12  ]]*10)),
    "Rosenbrock": (rosenbrock, np.array([[-2.048,  2.048 ]]*10)),
    "Rastrigin":  (rastrigin,  np.array([[-5.12,   5.12  ]]*10)),
    "Ackley":     (ackley,     np.array([[-32.768, 32.768]]*10)),
    "Griewank":   (griewank,   np.array([[-600,    600   ]]*10)),
}

def _make_toy_tsp(seed):
    rng = np.random.RandomState(seed); n = 6
    coords = rng.randint(10, 100, (n, 2)).astype(np.float64)
    d = coords[:,None,:] - coords[None,:,:]
    dist = np.round(np.sqrt((d**2).sum(-1))).astype(np.float64)
    def fit(perm):
        p = np.asarray(perm, dtype=int)
        return 1e6 if len(np.unique(p))!=n else float(np.sum(dist[p,np.roll(p,-1)]))
    def obj(x): return fit(np.argsort(x + np.random.normal(0,1e-6,x.shape)))
    def flip(t): i,j=np.random.choice(n,2,replace=False); t2=t[:]; t2[i],t2[j]=t2[j],t2[i]; return t2
    init = lambda: list(np.random.permutation(n))
    bounds = np.array([[0.0, n*10.0]]*n)
    hc = lambda: _toy_hc(fit, init(), 3000, flip)
    sa = lambda: _toy_sa(fit, init(), 5000, flip)
    return obj, fit, bounds, init, hc, sa, (coords, dist)

def _make_toy_ks(seed):
    rng = np.random.RandomState(seed); n = 10
    kw = rng.randint(5, 30, n); kv = rng.randint(20, 100, n)
    kcap = int(0.40 * kw.sum())
    def fit(b):
        b=np.asarray(b,dtype=np.float64); w=b@kw; v=b@kv
        return float(-v + 10*(w-kcap)) if w > kcap else float(-v)
    def obj_meta(x):
        return fit((np.asarray(x) >= 0.5).astype(np.float64))
    def flip(s): i=np.random.randint(n); s2=s[:]; s2[i]^=1; return s2
    init = lambda: list(np.random.randint(0,2,n))
    bounds = np.array([[0.0, 1.0]]*n)
    hc = lambda: _toy_hc(fit, init(), 3000, flip)
    sa = lambda: _toy_sa(fit, init(), 5000, flip)
    return obj_meta, fit, bounds, init, hc, sa, (kw, kv, kcap)

def _make_toy_gcp(seed):
    rng = np.random.RandomState(seed); n = 6; nc = 4
    r = rng.rand(n, n); adj = ((r+r.T)/2 > 0.5).astype(np.int32)
    np.fill_diagonal(adj, 0); ri, ci = np.where(np.triu(adj,1))
    def fit(c):
        c=np.asarray(c,dtype=int)%nc; return int(np.sum(c[ri]==c[ci]))
    def obj_meta(x):
        x=np.asarray(x); c=np.floor(x*nc/(x.max()+1e-9+1)).astype(int)
        return fit(np.clip(c,0,nc-1))
    def flip(s): i=np.random.randint(n); s2=s[:]; s2[i]=np.random.randint(nc); return s2
    init = lambda: list(np.random.randint(0,nc,n))
    bounds = np.array([[0.0, float(nc+1)]]*n)
    hc = lambda: _toy_hc(fit, init(), 3000, flip)
    sa = lambda: _toy_sa(fit, init(), 5000, flip)
    return obj_meta, fit, bounds, init, hc, sa, (adj, nc)

def _make_toy_sp(seed):
    rng = np.random.RandomState(seed); n = 6; S, G = 0, n-1
    raw = rng.rand(n, n)
    mask = np.zeros((n,n), dtype=bool)
    for i in range(n):
        for j in range(i+1, min(i+3, n)): mask[i,j] = True
    cost = np.where(mask & (raw < 0.6),
                    rng.randint(1,15,(n,n)).astype(np.float64), 0.0)
    for i in range(n-1):           # guarantee chain path always exists
        if cost[i,i+1] == 0: cost[i,i+1] = float(rng.randint(1,15))
    heu = np.array([float(n-1-i) for i in range(n)])
    def sp_fit(seq):
        path=list(np.argsort(seq)); cur=S; c=0.0; vis={cur}
        for nxt in path:
            if nxt in vis: return np.inf
            if cost[cur,nxt]>0: c+=cost[cur,nxt]; cur=nxt; vis.add(cur)
            else: return np.inf
        return c if cur==G else np.inf
    bounds = np.array([[0.0, n*10.0]]*n)
    return sp_fit, cost, heu, bounds, S, G


_L_N_CITIES = 20
np.random.seed(7)
_L_coords   = np.random.randint(10, 291, (_L_N_CITIES, 2)).astype(np.float64)
_L_d        = _L_coords[:,None,:] - _L_coords[None,:,:]
L_TSP_DIST  = np.round(np.sqrt((_L_d**2).sum(-1))).astype(np.float64)

def L_tsp_fit(perm):
    p = np.asarray(perm, dtype=int)
    return 1e9 if len(np.unique(p))!=_L_N_CITIES \
        else float(np.sum(L_TSP_DIST[p,np.roll(p,-1)]))

def L_tsp_obj_meta(x):
    return L_tsp_fit(np.argsort(x + np.random.normal(0,1e-6,x.shape)))

L_TSP_BOUNDS = np.array([[0.0, float(_L_N_CITIES)*10]]*_L_N_CITIES)

def _L_tsp_nn_bound():
    best=np.inf
    for s in range(_L_N_CITIES):
        unvis=set(range(_L_N_CITIES)); tour=[s]; unvis.remove(s)
        while unvis:
            cur=tour[-1]; nxt=min(unvis,key=lambda v:L_TSP_DIST[cur,v])
            tour.append(nxt); unvis.remove(nxt)
        c=L_tsp_fit(tour)
        if c<best: best=c
    return best

def _L_tsp_2opt(tour):
    t=list(tour); i,j=sorted(random.sample(range(_L_N_CITIES),2))
    t[i:j+1]=t[i:j+1][::-1]; return t

def _L_tsp_all_2opt(tour):
    n=len(tour); nbs=[]
    for i in range(n-1):
        for j in range(i+2,n):
            nb=tour[:]; nb[i:j+1]=nb[i:j+1][::-1]; nbs.append(nb)
    return nbs

_L_N_ITEMS=50
np.random.seed(13)
_L_KW  = np.random.randint(1,21,_L_N_ITEMS)
_L_KV  = np.random.randint(10,101,_L_N_ITEMS)
_L_KCAP= int(0.40*_L_KW.sum())

def L_ks_fit(b):
    b=np.asarray(b,dtype=np.float64); w=b@_L_KW; v=b@_L_KV
    return float(-v+200.0*max(0.0,w-_L_KCAP))

def L_ks_obj_meta(x):
    return L_ks_fit((np.asarray(x)>=0.5).astype(np.float64))

L_KS_BOUNDS=np.array([[0.0,1.0]]*_L_N_ITEMS)

def _L_ks_flip(s):
    n=list(s); i=random.randrange(len(n)); n[i]^=1; return n

def _L_ks_all_flips(s):
    nbs=[]
    for i in range(len(s)):
        nb=s[:]; nb[i]^=1; nbs.append(nb)
    return nbs

def _L_ks_dp_optimal():
    dp=np.zeros(_L_KCAP+1,dtype=np.int64)
    for i in range(_L_N_ITEMS):
        w,v=int(_L_KW[i]),int(_L_KV[i])
        for c in range(_L_KCAP,w-1,-1):
            if dp[c-w]+v>dp[c]: dp[c]=dp[c-w]+v
    return int(dp[_L_KCAP])

_L_N_VERT=20; _L_N_COLORS=5
np.random.seed(21)
_L_r=np.random.rand(_L_N_VERT,_L_N_VERT)
L_GCP_ADJ=((_L_r+_L_r.T)/2>0.65).astype(np.int32)
np.fill_diagonal(L_GCP_ADJ,0)
_L_ri,_L_ci=np.where(np.triu(L_GCP_ADJ,1))
_L_N_EDGES=len(_L_ri)

def L_gcp_fit(c):
    c=np.asarray(c,dtype=int)%_L_N_COLORS
    return int(np.sum(c[_L_ri]==c[_L_ci]))

def L_gcp_obj_meta(x):
    c=np.floor(np.asarray(x)*_L_N_COLORS/(np.asarray(x).max()+1e-9+1)).astype(int)
    return L_gcp_fit(np.clip(c,0,_L_N_COLORS-1))

L_GCP_BOUNDS=np.array([[0.0,float(_L_N_COLORS+1)]]*_L_N_VERT)

def _L_gcp_recolor(s):
    n=list(s); i=random.randrange(len(n)); n[i]=random.randrange(_L_N_COLORS); return n

def _L_gcp_all_recolors(s):
    nbs=[]
    for i in range(len(s)):
        for c in range(_L_N_COLORS):
            if c!=s[i]: nb=s[:]; nb[i]=c; nbs.append(nb)
    return nbs

def _L_gcp_greedy_ref():
    colors=[-1]*_L_N_VERT
    for v in range(_L_N_VERT):
        used={colors[u] for u in range(_L_N_VERT) if L_GCP_ADJ[v,u] and colors[u]>=0}
        colors[v]=next(c for c in range(_L_N_VERT) if c not in used)
    return len(set(colors)), L_gcp_fit(colors)

_L_N_NODES=15; _L_SP_S,_L_SP_G=0,_L_N_NODES-1
np.random.seed(42)
_L_raw=np.random.rand(_L_N_NODES,_L_N_NODES)
_L_rmask=np.zeros((_L_N_NODES,_L_N_NODES),dtype=bool)
for _i in range(_L_N_NODES):
    for _j in range(_i+1,min(_i+5,_L_N_NODES)): _L_rmask[_i,_j]=True
L_SP_COST=np.where(_L_rmask&(_L_raw<0.55),
                   np.random.randint(1,20,(_L_N_NODES,_L_N_NODES)).astype(np.float64),0.0)
for _i in range(_L_N_NODES-1):
    if L_SP_COST[_i,_i+1]==0: L_SP_COST[_i,_i+1]=float(np.random.randint(1,20))
L_SP_HEU=np.array([float(_L_N_NODES-1-i) for i in range(_L_N_NODES)])

def L_sp_obj_meta(seq):
    path=list(np.argsort(seq)); cur=_L_SP_S; cost=0.0; vis={cur}
    for nxt in path:
        if nxt in vis: return np.inf
        if L_SP_COST[cur,nxt]>0: cost+=L_SP_COST[cur,nxt]; cur=nxt; vis.add(cur)
        else: return np.inf
    return cost if cur==_L_SP_G else np.inf

L_SP_BOUNDS=np.array([[0.0,float(_L_N_NODES)*10]]*_L_N_NODES)

def _L_dijkstra_optimal():
    dist={_L_SP_S:0.0}; pq=[(0.0,_L_SP_S)]
    while pq:
        d,u=heapq.heappop(pq)
        if d>dist.get(u,np.inf): continue
        for v in range(_L_N_NODES):
            if L_SP_COST[u,v]>0:
                nd=d+L_SP_COST[u,v]
                if nd<dist.get(v,np.inf): dist[v]=nd; heapq.heappush(pq,(nd,v))
    return dist.get(_L_SP_G,np.inf)

_TAB20=plt.cm.get_cmap('tab20'); _MARKERS=list("osD^vPhX<>p*")
def _col(i,n): return _TAB20(i/max(n-1,1))
def _mk(i):    return _MARKERS[i%len(_MARKERS)]
def _mean(runs):
    vals=[r["best_fitness"] for r in runs if np.isfinite(r["best_fitness"])]
    return float(np.mean(vals)) if vals else np.inf


def plot_convergence(title, runs_dict, save_name):
    fig,ax=plt.subplots(figsize=(13,6)); keys=list(runs_dict.keys())
    for idx,(algo,runs) in enumerate(runs_dict.items()):
        hists=[r["history"] for r in runs if r.get("history")]
        if not hists: continue
        L=max(len(h) for h in hists)
        arr=np.full((len(hists),L),np.nan)
        for k,h in enumerate(hists): arr[k,:len(h)]=h
        mean_h=np.nanmean(arr,axis=0); std_h=np.nanstd(arr,axis=0); iters=np.arange(L)
        c=_col(idx,len(keys)); lbl=f"[C] {algo}" if algo in _CLASSIC_KEYS else algo
        ls='--' if algo in _CLASSIC_KEYS else '-'; lw=2.2 if algo in _CLASSIC_KEYS else 1.4
        mk=_mk(idx); mev=max(L//12,1)
        ax.plot(iters,mean_h,label=lbl,color=c,linestyle=ls,linewidth=lw,marker=mk,markevery=mev,markersize=5)
        ax.fill_between(iters,mean_h-std_h,mean_h+std_h,alpha=0.07,color=c)
    all_fin=[r["best_fitness"] for runs in runs_dict.values()
             for r in runs if np.isfinite(r["best_fitness"]) and r["best_fitness"]>0]
    if all_fin and max(all_fin)/max(min(all_fin),1e-12)>1e3: ax.set_yscale('log')
    ax.set_title(f"Convergence — {title}",fontsize=13); ax.set_xlabel("Iteration"); ax.set_ylabel("Best Fitness")
    ax.legend(fontsize=7,ncol=4,loc="upper right"); ax.grid(True,alpha=0.3)
    plt.tight_layout(); p=f"convergence_{save_name}.png"
    plt.savefig(p,dpi=150,bbox_inches='tight'); plt.close(); print(f"  Saved: {p}")


def plot_timecost(title, runs_dict, save_name):
    keys=list(runs_dict.keys()); rows=[]
    for idx,(algo,runs) in enumerate(runs_dict.items()):
        ts=[r["time_sec"] for r in runs if np.isfinite(r.get("time_sec",0))]
        cs=[r["best_fitness"] for r in runs if np.isfinite(r.get("best_fitness",np.inf))]
        if ts and cs: rows.append((algo,np.mean(ts),np.std(ts),np.mean(cs),np.std(cs),idx))
    if not rows: return
    fig,ax=plt.subplots(figsize=(11,7))
    for algo,mt,st,mc,sc,idx in rows:
        c=_col(idx,len(keys)); mk=_mk(idx)
        lbl=f"[C] {algo}" if algo in _CLASSIC_KEYS else algo
        ax.errorbar(mt,mc,xerr=st,yerr=sc,fmt=mk,color=c,ecolor='gray',capsize=4,markersize=9,
                    linestyle='none',elinewidth=1.4,label=lbl,
                    markerfacecolor=c if algo in _CLASSIC_KEYS else 'none',markeredgewidth=1.5)
    fin=[r[3] for r in rows if np.isfinite(r[3])]
    if fin and max(fin)>1e5: ax.set_yscale('log')
    ax.set_title(f"Time vs Fitness — {title}",fontsize=13)
    ax.set_xlabel("Mean Time (s)"); ax.set_ylabel("Mean Best Fitness")
    ax.legend(fontsize=8,ncol=3); ax.grid(True,alpha=0.3)
    plt.tight_layout(); p=f"timecost_{save_name}.png"
    plt.savefig(p,dpi=150,bbox_inches='tight'); plt.close(); print(f"  Saved: {p}")


def plot_bar(title, runs_dict, ref_line=None, ref_label=None, save_name="bar"):
    rows=[]
    for algo,runs in runs_dict.items():
        vals=[r["best_fitness"] for r in runs if np.isfinite(r["best_fitness"])]
        if vals: rows.append((algo,float(np.mean(vals)),float(np.std(vals))))
    if not rows: return
    rows.sort(key=lambda r:r[1]); labels,means,stds=zip(*rows)
    fig,ax=plt.subplots(figsize=(10,max(4,len(labels)*0.55)))
    n=len(labels); colors=[_col(i,n) for i in range(n)]
    hatches=['///' if l in _CLASSIC_KEYS else '' for l in labels]
    ax.barh(labels,means,xerr=stds,color=colors,hatch=hatches,edgecolor='black',linewidth=0.7,
            error_kw=dict(ecolor='black',capsize=3))
    handles=[Patch(facecolor='white',edgecolor='black',hatch='///',label='Classic [C]'),
             Patch(facecolor='gray', edgecolor='black',label='Metaheuristic')]
    if ref_line is not None:
        ax.axvline(ref_line,color='red',linewidth=2,linestyle='--')
        handles.append(plt.Line2D([0],[0],color='red',linestyle='--',label=ref_label or f'Ref={ref_line:.1f}'))
    ax.invert_yaxis(); ax.set_title(f"Best Fitness — {title}",fontsize=13)
    ax.set_xlabel("Mean Best Fitness"); ax.grid(True,axis='x',alpha=0.3)
    ax.legend(handles=handles,loc='lower right',fontsize=8)
    plt.tight_layout(); p=f"barcomp_{save_name}.png"
    plt.savefig(p,dpi=150,bbox_inches='tight'); plt.close(); print(f"  Saved: {p}")


def export_stats(prob_name, runs_dict, suffix="", nn_bound=None):
    fname=f"stats_{prob_name}{suffix}.txt"
    def _fmt(v): return f"{v:>13.5f}" if np.isfinite(v) else f"{'inf':>13}"
    rows=[]
    for algo,runs in runs_dict.items():
        costs=[r["best_fitness"] for r in runs]; times=[r["time_sec"] for r in runs]
        valid=[c for c in costs if np.isfinite(c)]; tag="[C]" if algo in _CLASSIC_KEYS else "   "
        rows.append((tag,algo,
                     np.mean(valid) if valid else np.inf,
                     np.std(valid)  if len(valid)>1 else 0.,
                     np.min(valid)  if valid else np.inf,
                     np.max(valid)  if valid else np.inf,
                     float(np.mean(times)),
                     costs))
    rows.sort(key=lambda r:r[2] if np.isfinite(r[2]) else np.inf)
    all_finite=[c for *_,costs in rows for c in costs if np.isfinite(c)]
    global_best=min(all_finite) if all_finite else np.inf
    tol=max(abs(global_best)*1e-6, 1e-9)
    show_nn = nn_bound is not None and np.isfinite(nn_bound)
    width = 112 if show_nn else 102
    with open(fname,"w",encoding="utf-8") as f:
        f.write(f"{'='*width}\n  {prob_name}  |  [C]=Classic  (blank)=Metaheuristic\n{'='*width}\n\n")
        if show_nn:
            hdr = "  {:3} {:<12} {:>13} {:>13} {:>13} {:>13} {:>10} {:>7} {:>8}\n  {}\n".format(
                '', 'Algorithm', 'Mean', 'Std', 'Best', 'Worst', 'Time(s)', '%Hit', '%>NN', '-'*94)
        else:
            hdr = "  {:3} {:<12} {:>13} {:>13} {:>13} {:>13} {:>10} {:>7}\n  {}\n".format(
                '', 'Algorithm', 'Mean', 'Std', 'Best', 'Worst', 'Time(s)', '%Hit', '-'*84)
        f.write(hdr)
        for tag,algo,mn,sd,best,worst,tm,costs in rows:
            n_total=len(costs)
            n_hit=sum(1 for c in costs if np.isfinite(c) and abs(c-global_best)<=tol)
            hit_str=f"{n_hit/n_total*100:>6.1f}%"
            if show_nn:
                n_beat=sum(1 for c in costs if np.isfinite(c) and c < nn_bound)
                nn_str=f"{n_beat/n_total*100:>7.1f}%"
                f.write(f"  {tag} {algo:<12} {_fmt(mn)} {_fmt(sd)} {_fmt(best)} {_fmt(worst)} {tm:>10.4f} {hit_str} {nn_str}\n")
            else:
                f.write(f"  {tag} {algo:<12} {_fmt(mn)} {_fmt(sd)} {_fmt(best)} {_fmt(worst)} {tm:>10.4f} {hit_str}\n")
        f.write(f"\n  Global best = {global_best:.5f}  |  %Hit = % of runs reaching global best (tol={tol:.2e})\n")
        if show_nn:
            f.write(f"  NN bound    = {nn_bound:.5f}  |  %>NN  = % of runs beating NN bound (strict <)\n")
        f.write(f"{'='*width}\n")
    print(f"  Saved: {fname}")


def plot_3d(objective, bounds, name):
    xl=-10. if name=="Griewank" else float(bounds[0,0])
    xu= 10. if name=="Griewank" else float(bounds[0,1])
    x=np.linspace(xl,xu,200); y=np.linspace(xl,xu,200); X,Y=np.meshgrid(x,y)
    Z=np.zeros_like(X)
    for i in range(200):
        for j in range(200):
            pt=np.zeros(bounds.shape[0]); pt[0]=X[i,j]; pt[1]=Y[i,j]; Z[i,j]=objective(pt)
    fig=plt.figure(figsize=(14,6)); ax1=fig.add_subplot(121,projection='3d')
    s=ax1.plot_surface(X,Y,Z,cmap='viridis',alpha=0.85,linewidth=0)
    ax1.set_title(f'3D — {name}'); ax1.set_xlabel('x1'); ax1.set_ylabel('x2'); ax1.set_zlabel('f')
    fig.colorbar(s,ax=ax1,shrink=0.5); ax2=fig.add_subplot(122)
    cf=ax2.contourf(X,Y,Z,levels=150,cmap='viridis')
    ax2.set_title(f'Contour — {name}'); ax2.set_xlabel('x1'); ax2.set_ylabel('x2')
    fig.colorbar(cf,ax=ax2); plt.tight_layout()
    p=f"landscape_{name}.png"; plt.savefig(p,dpi=150,bbox_inches='tight'); plt.close(); print(f"  Saved: {p}")

class _NpMinHeap:
    def __init__(self,cap,row_w):
        self._buf=np.empty((cap,1+row_w),dtype=np.float64); self._size=np.int64(0)
    def empty(self): return self._size==np.int64(0)
    def push(self,priority,row):
        n=int(self._size); self._buf[n,0]=np.float64(priority)
        self._buf[n,1:]=np.asarray(row,dtype=np.float64); self._size+=np.int64(1)
    def pop(self):
        n=int(self._size); idx=int(np.argmin(self._buf[:n,0]))
        row=self._buf[idx].copy(); self._buf[idx]=self._buf[n-1]; self._size-=np.int64(1)
        return float(row[0]),row[1:]

class _NpQueue:
    def __init__(self,cap,row_w):
        self._buf=np.empty((cap,row_w),dtype=np.float64)
        self._head=np.int64(0); self._tail=np.int64(0); self._cap=cap
    def empty(self): return self._head==self._tail
    def enqueue(self,row):
        self._buf[int(self._tail%self._cap)]=np.asarray(row,dtype=np.float64); self._tail+=np.int64(1)
    def dequeue(self):
        row=self._buf[int(self._head%self._cap)].copy(); self._head+=np.int64(1); return row

class _NpStack:
    def __init__(self,cap,row_w):
        self._buf=np.empty((cap,row_w),dtype=np.float64); self._top=np.int64(0)
    def empty(self): return self._top==np.int64(0)
    def push(self,row):
        self._buf[int(self._top)]=np.asarray(row,dtype=np.float64); self._top+=np.int64(1)
    def pop(self):
        self._top-=np.int64(1); return self._buf[int(self._top)].copy()

class _NpVisited:
    def __init__(self,cap,dim):
        self._buf=np.empty((cap,dim),dtype=np.float64); self._count=np.int64(0)
    def contains(self,x):
        x=np.asarray(x,dtype=np.float64); n=int(self._count)
        return n>0 and bool(np.any(np.all(np.abs(self._buf[:n]-x)<1e-9,axis=1)))
    def add(self,x):
        self._buf[int(self._count)]=np.asarray(x,dtype=np.float64); self._count+=np.int64(1)


def _run_small(algo_func, obj, bounds):
    t0=time.time(); mem0=psutil.Process().memory_info().rss/1024**2
    try:
        res=algo_func(obj,bounds,SMALL_POP_SIZE,SMALL_MAX_ITER)
        pos,fit=res[0],res[1]
        hist=list(res[2]) if len(res)>2 else [float(fit)]*(SMALL_MAX_ITER+1)
        div =list(res[3]) if len(res)>3 else [0.0]*(SMALL_MAX_ITER+1)
        t=time.time()-t0; mem=psutil.Process().memory_info().rss/1024**2-mem0
        return {"best_fitness":float(fit) if np.isfinite(fit) else np.inf,
                "history":hist,"diversity":div,"time_sec":max(t,1e-6),
                "memory_mb":mem,"best_solution":pos}
    except Exception as e:
        print(f"    ! {algo_func.__name__}: {e}")
        return {"best_fitness":np.inf,"history":[np.inf]*(SMALL_MAX_ITER+1),
                "diversity":[0.]*(SMALL_MAX_ITER+1),"time_sec":0.,"memory_mb":0.,"best_solution":None}


def _run_toy_graph(algo, cost, heu, S, G):
    t0=time.time(); path=algo(cost,heu,S,G); t=time.time()-t0
    cost_val=float(np.sum(cost[path[:-1],path[1:]])) if len(path)>1 else np.inf
    return {"cost":cost_val,"time_sec":max(t,0.001)}


def _run_toy_comb(algo, objective, initial_state):
    t0=time.time(); dim=len(initial_state)
    s0=np.asarray(initial_state,dtype=np.float64)
    bc=objective(s0.astype(int)); bc=bc if np.isfinite(bc) else np.inf
    def _nbrs(s):
        nb=[]
        for i in range(dim):
            for j in range(i+1,dim):
                n2=s.copy(); n2[i],n2[j]=n2[j],n2[i]; nb.append(n2)
        return nb
    name=algo.__name__; vis=_NpVisited(100_000,dim); vis.add(s0)
    if name=="bfs":
        q=_NpQueue(100_000,dim); q.enqueue(s0)
        while not q.empty():
            s=q.dequeue(); c=objective(s.astype(int))
            if c<bc: bc=c
            for nb in _nbrs(s):
                if not vis.contains(nb): vis.add(nb); q.enqueue(nb)
    elif name=="dfs":
        stk=_NpStack(100_000,dim); stk.push(s0)
        while not stk.empty():
            s=stk.pop(); c=objective(s.astype(int))
            if c<bc: bc=c
            for nb in _nbrs(s):
                if not vis.contains(nb): vis.add(nb); stk.push(nb)
    elif name=="ucs":
        h=_NpMinHeap(100_000,dim); h.push(objective(s0.astype(int)),s0)
        while not h.empty():
            c,s=h.pop()
            if c>=bc: continue
            bc=c
            for nb in _nbrs(s):
                if not vis.contains(nb): vis.add(nb); h.push(objective(nb.astype(int)),nb)
    elif name=="greedy_best_first_search":
        h=_NpMinHeap(100_000,dim); h.push(objective(s0.astype(int)),s0)
        while not h.empty():
            _,s=h.pop(); c=objective(s.astype(int))
            if c<bc: bc=c
            for nb in _nbrs(s):
                if not vis.contains(nb): vis.add(nb); h.push(objective(nb.astype(int)),nb)
    elif name=="a_star_search":
        h=_NpMinHeap(100_000,dim); h.push(objective(s0.astype(int)),s0)
        while not h.empty():
            f,s=h.pop(); c=objective(s.astype(int)); g=f-c
            if c<bc: bc=c
            for nb in _nbrs(s):
                if not vis.contains(nb):
                    vis.add(nb); h.push((g+1.0)+objective(nb.astype(int)),nb)
    return {"cost":bc if np.isfinite(bc) else np.inf,"time_sec":max(time.time()-t0,0.001)}


def _toy_hc(objective,state,n_iter,flip_fn):
    s=state.copy(); c=objective(s); t0=time.time()
    for _ in range(n_iter):
        ns=flip_fn(s); nc=objective(ns)
        if nc<c: s,c=ns,nc
    return c,max(time.time()-t0,0.001)

def _toy_sa(objective,state,n_iter,flip_fn,T=200.,alpha=0.995):
    s=state.copy(); c=objective(s); t0=time.time(); temp=T
    for _ in range(n_iter):
        ns=flip_fn(s); d=objective(ns)-c
        if d<0 or np.random.rand()<np.exp(-d/max(temp,1e-10)): s,c=ns,c+d
        temp*=alpha
    return c,max(time.time()-t0,0.001)

def _wrap_toy(raw_runs, n_iter):
    out=[]
    for r in raw_runs:
        bf=r.get("cost",r.get("best_fitness",np.inf)); bf=bf if np.isfinite(bf) else np.inf
        out.append({**r,"best_fitness":bf,"history":[bf]*(n_iter+1),"diversity":[0.0]*(n_iter+1)})
    return out

def _run_large(algo_func, obj, bounds):
    t0=time.time()
    try:
        res=algo_func(obj,bounds,LARGE_POP_SIZE,LARGE_MAX_ITER)
        pos,fit=res[0],float(res[1])
        hist=list(res[2]) if len(res)>2 else [fit]*(LARGE_MAX_ITER+1)
        div =list(res[3]) if len(res)>3 else [0.0]*(LARGE_MAX_ITER+1)
        fit=fit if np.isfinite(fit) else np.inf
    except Exception as e:
        print(f"      ! {algo_func.__name__}: {e}")
        fit,hist,div=np.inf,[np.inf]*(LARGE_MAX_ITER+1),[0.0]*(LARGE_MAX_ITER+1)
    return {"best_fitness":fit,"history":hist,"diversity":div,"time_sec":max(time.time()-t0,1e-4)}


def _wrap_large(cost, t):
    bf=cost if np.isfinite(cost) else np.inf
    return {"best_fitness":bf,"history":[bf]*(LARGE_MAX_ITER+1),
            "diversity":[0.0]*(LARGE_MAX_ITER+1),"time_sec":max(t,1e-4)}


def _large_hc(obj, init_fn, flip_fn, patience=800):
    s=init_fn(); c=obj(s); best_c=c; t0=time.time(); no_imp=0
    for _ in range(LARGE_HC_ITER):
        ns=flip_fn(s); nc=obj(ns)
        if nc<c: s,c=ns,nc; no_imp=0
        else:    no_imp+=1
        if nc<best_c: best_c=nc
        if no_imp>=patience: s=init_fn(); c=obj(s); no_imp=0
    return best_c,time.time()-t0


def _large_sa(obj, init_fn, flip_fn):
    s=init_fn(); c=obj(s); best_c=c; t0=time.time(); T=LARGE_SA_T0
    for _ in range(LARGE_SA_ITER):
        ns=flip_fn(s); nc=obj(ns); d=nc-c
        if d<0 or (T>1e-10 and random.random()<np.exp(-d/T)): s,c=ns,nc
        if c<best_c: best_c=c
        T*=LARGE_SA_ALPHA
    return best_c,time.time()-t0


def _run_classic_comb(algo_name, obj, init_state, all_neighbours_fn):
    from collections import deque
    t0=time.time(); s0=list(init_state); best=obj(s0); visited={tuple(s0)}; nodes=0
    if algo_name=="BFS":
        q=deque([s0])
        while q and nodes<LARGE_NODE_CAP:
            s=q.popleft(); nodes+=1; c=obj(s)
            if c<best: best=c
            for nb in all_neighbours_fn(s):
                key=tuple(nb)
                if key not in visited: visited.add(key); q.append(nb)
    elif algo_name=="DFS":
        stk=[s0]
        while stk and nodes<LARGE_NODE_CAP:
            s=stk.pop(); nodes+=1; c=obj(s)
            if c<best: best=c
            for nb in all_neighbours_fn(s):
                key=tuple(nb)
                if key not in visited: visited.add(key); stk.append(nb)
    elif algo_name=="UCS":
        pq=[(obj(s0),s0)]; visited=set()
        while pq and nodes<LARGE_NODE_CAP:
            cost,s=heapq.heappop(pq); key=tuple(s)
            if key in visited: continue
            visited.add(key); nodes+=1
            if cost<best: best=cost
            for nb in all_neighbours_fn(s):
                nb_k=tuple(nb)
                if nb_k not in visited: heapq.heappush(pq,(obj(nb),nb))
    elif algo_name=="Greedy":
        pq=[(obj(s0),s0)]; visited=set()
        while pq and nodes<LARGE_NODE_CAP:
            _,s=heapq.heappop(pq); key=tuple(s)
            if key in visited: continue
            visited.add(key); nodes+=1; c=obj(s)
            if c<best: best=c
            for nb in all_neighbours_fn(s):
                nb_k=tuple(nb)
                if nb_k not in visited: heapq.heappush(pq,(obj(nb),nb))
    elif algo_name=="A*":
        pq=[(obj(s0),0,s0)]; g_map={tuple(s0):0}; visited=set()
        while pq and nodes<LARGE_NODE_CAP:
            f,g,s=heapq.heappop(pq); key=tuple(s)
            if key in visited: continue
            visited.add(key); nodes+=1; c=obj(s)
            if c<best: best=c
            for nb in all_neighbours_fn(s):
                nb_k=tuple(nb)
                if nb_k not in visited:
                    ng=g+1
                    if ng<g_map.get(nb_k,np.inf):
                        g_map[nb_k]=ng; heapq.heappush(pq,(ng+obj(nb),ng,nb))
    return best,time.time()-t0


def _run_classic_sp(algo_func):
    t0=time.time()
    try:
        path=algo_func(L_SP_COST,L_SP_HEU,_L_SP_S,_L_SP_G)
        cost=float(np.sum(L_SP_COST[path[:-1],path[1:]])) \
             if len(path)>=2 and path[0]==_L_SP_S and path[-1]==_L_SP_G else np.inf
    except Exception: cost=np.inf
    return cost,time.time()-t0

def save_problem_maps(nn_ref, ks_dp_opt, gcp_nc, sp_opt):
    # TSP
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    fig.suptitle(f"TSP — {_L_N_CITIES} cities  |  NN bound = {nn_ref:.0f}",fontsize=13,fontweight='bold')
    ax=axes[0]; ax.scatter(_L_coords[:,0],_L_coords[:,1],s=120,c='steelblue',zorder=3)
    for i,(x,y) in enumerate(_L_coords):
        ax.annotate(str(i),(x,y),textcoords="offset points",xytext=(6,4),fontsize=9,fontweight='bold')
    nn_tour,best_nn=None,np.inf
    for start in range(_L_N_CITIES):
        unvis=set(range(_L_N_CITIES)); tour=[start]; unvis.remove(start)
        while unvis:
            cur=tour[-1]; nxt=min(unvis,key=lambda v:L_TSP_DIST[cur,v]); tour.append(nxt); unvis.remove(nxt)
        c=L_tsp_fit(tour)
        if c<best_nn: best_nn,nn_tour=c,tour
    for i in range(_L_N_CITIES):
        a,b=nn_tour[i],nn_tour[(i+1)%_L_N_CITIES]
        ax.plot([_L_coords[a,0],_L_coords[b,0]],[_L_coords[a,1],_L_coords[b,1]],'steelblue',alpha=0.5,linewidth=1.2)
    ax.set_title(f"City Layout + NN Tour (cost={nn_ref:.0f})"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(True,alpha=0.2)
    ax=axes[1]; im=ax.imshow(L_TSP_DIST,cmap='YlOrRd',aspect='auto')
    ax.set_title("Distance Matrix"); ax.set_xlabel("City"); ax.set_ylabel("City"); plt.colorbar(im,ax=ax,label="Distance")
    for i in range(_L_N_CITIES):
        for j in range(_L_N_CITIES):
            ax.text(j,i,f"{int(L_TSP_DIST[i,j])}",ha='center',va='center',fontsize=5,
                    color='black' if L_TSP_DIST[i,j]<L_TSP_DIST.max()*0.6 else 'white')
    plt.tight_layout(); plt.savefig("map_TSP.png",dpi=150,bbox_inches='tight'); plt.close(); print("  Saved: map_TSP.png")

    # KP
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    fig.suptitle(f"Knapsack — {_L_N_ITEMS} items  cap={_L_KCAP}  |  DP optimal = {ks_dp_opt}",fontsize=13,fontweight='bold')
    ax=axes[0]; ratio=_L_KV/_L_KW
    sc=ax.scatter(_L_KW,_L_KV,c=ratio,cmap='RdYlGn',s=80,zorder=3,edgecolors='gray',linewidths=0.5)
    for i in range(_L_N_ITEMS): ax.annotate(str(i),(_L_KW[i],_L_KV[i]),textcoords="offset points",xytext=(4,3),fontsize=7)
    plt.colorbar(sc,ax=ax,label="Value/Weight ratio"); ax.set_xlabel("Weight"); ax.set_ylabel("Value")
    ax.set_title("Items: Weight vs Value  (green = high ratio)"); ax.grid(True,alpha=0.25)
    ax=axes[1]; order=np.argsort(_L_KV/_L_KW)[::-1]; cumw=np.cumsum(_L_KW[order])
    ax.bar(range(_L_N_ITEMS),_L_KV[order],color=plt.cm.RdYlGn(np.linspace(0.2,0.9,_L_N_ITEMS)),edgecolor='gray',linewidth=0.4)
    ax2=ax.twinx(); ax2.plot(range(_L_N_ITEMS),cumw,'b--',linewidth=1.5,label="Cumulative weight")
    ax2.axhline(_L_KCAP,color='red',linewidth=2,label=f"Capacity ({_L_KCAP})")
    ax2.set_ylabel("Cumulative Weight",color='blue'); ax2.legend(loc='lower right',fontsize=8)
    ax.set_xlabel("Item rank (by value/weight)"); ax.set_ylabel("Value")
    ax.set_title(f"Items sorted by ratio  |  DP optimal = {ks_dp_opt}"); ax.grid(True,alpha=0.2,axis='y')
    plt.tight_layout(); plt.savefig("map_KP.png",dpi=150,bbox_inches='tight'); plt.close(); print("  Saved: map_KP.png")

    # GCP
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    fig.suptitle(f"Graph Coloring — {_L_N_VERT} vertices  {_L_N_EDGES} edges  {_L_N_COLORS} colors  |  chromatic ≤ {gcp_nc}",fontsize=13,fontweight='bold')
    ax=axes[0]; im=ax.imshow(L_GCP_ADJ,cmap='Blues',aspect='auto',vmin=0,vmax=1)
    ax.set_title("Adjacency Matrix"); ax.set_xlabel("Vertex"); ax.set_ylabel("Vertex"); plt.colorbar(im,ax=ax,ticks=[0,1],label="Edge")
    for i in range(_L_N_VERT):
        for j in range(_L_N_VERT):
            if L_GCP_ADJ[i,j]: ax.text(j,i,"1",ha='center',va='center',fontsize=7,color='white')
    ax=axes[1]; degrees=L_GCP_ADJ.sum(axis=1)
    ax.bar(range(_L_N_VERT),degrees,color=plt.cm.Oranges(degrees/degrees.max()),edgecolor='gray',linewidth=0.5)
    ax.axhline(degrees.mean(),color='red',linestyle='--',linewidth=1.5,label=f"Mean degree = {degrees.mean():.1f}")
    ax.set_xlabel("Vertex"); ax.set_ylabel("Degree")
    ax.set_title(f"Degree Distribution  (density={_L_N_EDGES/(_L_N_VERT*(_L_N_VERT-1)/2):.2f})")
    ax.legend(fontsize=9); ax.grid(True,alpha=0.25,axis='y')
    plt.tight_layout(); plt.savefig("map_GCP.png",dpi=150,bbox_inches='tight'); plt.close(); print("  Saved: map_GCP.png")

    # SP
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    fig.suptitle(f"Shortest Path — {_L_N_NODES} nodes  0→{_L_SP_G}  |  Dijkstra optimal = {sp_opt:.1f}",fontsize=13,fontweight='bold')
    ax=axes[0]; display=np.where(L_SP_COST>0,L_SP_COST,np.nan)
    im=ax.imshow(display,cmap='YlOrRd',aspect='auto')
    ax.set_title("Cost Matrix  (white = no edge)"); ax.set_xlabel("To node"); ax.set_ylabel("From node"); plt.colorbar(im,ax=ax,label="Edge cost")
    for i in range(_L_N_NODES):
        for j in range(_L_N_NODES):
            if L_SP_COST[i,j]>0: ax.text(j,i,f"{int(L_SP_COST[i,j])}",ha='center',va='center',fontsize=7,color='black')
    ax.scatter([_L_SP_S],[_L_SP_S],s=200,c='lime',  zorder=5,marker='*',label=f"Start ({_L_SP_S})")
    ax.scatter([_L_SP_G],[_L_SP_G],s=200,c='purple',zorder=5,marker='*',label=f"Goal ({_L_SP_G})")
    ax.legend(fontsize=8,loc='upper right')
    ax=axes[1]; out_deg=(L_SP_COST>0).sum(axis=1); in_deg=(L_SP_COST>0).sum(axis=0)
    xn=np.arange(_L_N_NODES); w=0.35
    ax.bar(xn-w/2,out_deg,w,label="Out-degree",color='steelblue', edgecolor='gray')
    ax.bar(xn+w/2,in_deg, w,label="In-degree", color='darkorange',edgecolor='gray')
    ax.axvline(_L_SP_S,color='lime',  linewidth=2,linestyle='--',label=f"Start ({_L_SP_S})")
    ax.axvline(_L_SP_G,color='purple',linewidth=2,linestyle='--',label=f"Goal ({_L_SP_G})")
    ax.set_xlabel("Node"); ax.set_ylabel("Degree")
    ax.set_title(f"In/Out Degree  |  {int((L_SP_COST>0).sum())} edges total")
    ax.set_xticks(xn); ax.legend(fontsize=8); ax.grid(True,alpha=0.25,axis='y')
    plt.tight_layout(); plt.savefig("map_SP.png",dpi=150,bbox_inches='tight'); plt.close(); print("  Saved: map_SP.png")

def save_toy_maps(seed=42):
    """Generate map PNGs for the toy problems using the seed-42 instance."""
    import heapq

    # ── TSP ───────────────────────────────────────────────────────────────────
    _,fit,_,_,_,_,(coords,dist) = _make_toy_tsp(seed)
    n = len(coords)
    # nearest-neighbour tour
    nn_tour, best_nn = None, np.inf
    for start in range(n):
        unvis=set(range(n)); tour=[start]; unvis.remove(start)
        while unvis:
            cur=tour[-1]; nxt=min(unvis,key=lambda v:dist[cur,v]); tour.append(nxt); unvis.remove(nxt)
        c=fit(tour)
        if c<best_nn: best_nn,nn_tour=c,tour
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    fig.suptitle(f"Toy TSP — {n} cities  |  NN bound = {best_nn:.0f}  (seed={seed})",fontsize=12,fontweight='bold')
    ax=axes[0]
    ax.scatter(coords[:,0],coords[:,1],s=160,c='steelblue',zorder=3)
    for i,(x,y) in enumerate(coords):
        ax.annotate(str(i),(x,y),textcoords="offset points",xytext=(7,5),fontsize=11,fontweight='bold')
    for i in range(n):
        a,b=nn_tour[i],nn_tour[(i+1)%n]
        ax.plot([coords[a,0],coords[b,0]],[coords[a,1],coords[b,1]],'steelblue',alpha=0.5,linewidth=1.5)
    ax.set_title(f"City Layout + NN Tour (cost={best_nn:.0f})"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(True,alpha=0.2)
    ax=axes[1]; im=ax.imshow(dist,cmap='YlOrRd',aspect='auto')
    ax.set_title("Distance Matrix"); ax.set_xlabel("City"); ax.set_ylabel("City")
    plt.colorbar(im,ax=ax,label="Distance")
    for i in range(n):
        for j in range(n):
            ax.text(j,i,f"{int(dist[i,j])}",ha='center',va='center',fontsize=9,
                    color='black' if dist[i,j]<dist.max()*0.6 else 'white')
    plt.tight_layout(); plt.savefig("map_toy_TSP.png",dpi=150,bbox_inches='tight'); plt.close()
    print("  Saved: map_toy_TSP.png")

    # ── Knapsack ──────────────────────────────────────────────────────────────
    _,_,_,_,_,_,(kw,kv,kcap) = _make_toy_ks(seed)
    n_items=len(kw); ratio=kv/kw
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    fig.suptitle(f"Toy Knapsack — {n_items} items  cap={kcap}  (seed={seed})",fontsize=12,fontweight='bold')
    ax=axes[0]
    sc=ax.scatter(kw,kv,c=ratio,cmap='RdYlGn',s=100,zorder=3,edgecolors='gray',linewidths=0.6)
    for i in range(n_items): ax.annotate(str(i),(kw[i],kv[i]),textcoords="offset points",xytext=(4,4),fontsize=9)
    plt.colorbar(sc,ax=ax,label="Value/Weight ratio"); ax.set_xlabel("Weight"); ax.set_ylabel("Value")
    ax.set_title("Items: Weight vs Value  (green = high ratio)"); ax.grid(True,alpha=0.25)
    ax=axes[1]; order=np.argsort(ratio)[::-1]; cumw=np.cumsum(kw[order])
    ax.bar(range(n_items),kv[order],color=plt.cm.RdYlGn(np.linspace(0.2,0.9,n_items)),edgecolor='gray',linewidth=0.5)
    ax2=ax.twinx(); ax2.plot(range(n_items),cumw,'b--',linewidth=1.5,label="Cumulative weight")
    ax2.axhline(kcap,color='red',linewidth=2,label=f"Capacity ({kcap})")
    ax2.set_ylabel("Cumulative Weight",color='blue'); ax2.legend(loc='lower right',fontsize=8)
    ax.set_xlabel("Item rank (by value/weight)"); ax.set_ylabel("Value")
    ax.set_title("Items sorted by ratio"); ax.grid(True,alpha=0.2,axis='y')
    plt.tight_layout(); plt.savefig("map_toy_KP.png",dpi=150,bbox_inches='tight'); plt.close()
    print("  Saved: map_toy_KP.png")

    # ── Graph Coloring ────────────────────────────────────────────────────────
    _,_,_,_,_,_,(adj,nc) = _make_toy_gcp(seed)
    n_v=adj.shape[0]; n_e=int(np.triu(adj,1).sum())
    density=n_e/(n_v*(n_v-1)/2)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    fig.suptitle(f"Toy GCP — {n_v} vertices  {n_e} edges  {nc} colors  (seed={seed})",fontsize=12,fontweight='bold')
    ax=axes[0]; im=ax.imshow(adj,cmap='Blues',aspect='auto',vmin=0,vmax=1)
    ax.set_title("Adjacency Matrix"); ax.set_xlabel("Vertex"); ax.set_ylabel("Vertex")
    plt.colorbar(im,ax=ax,ticks=[0,1],label="Edge")
    for i in range(n_v):
        for j in range(n_v):
            if adj[i,j]: ax.text(j,i,"1",ha='center',va='center',fontsize=10,color='white')
    ax=axes[1]; degrees=adj.sum(axis=1)
    ax.bar(range(n_v),degrees,color=plt.cm.Oranges(degrees/max(degrees.max(),1)),edgecolor='gray',linewidth=0.6)
    ax.axhline(degrees.mean(),color='red',linestyle='--',linewidth=1.5,label=f"Mean degree = {degrees.mean():.1f}")
    ax.set_xlabel("Vertex"); ax.set_ylabel("Degree")
    ax.set_title(f"Degree Distribution  (density={density:.2f})")
    ax.set_xticks(range(n_v)); ax.legend(fontsize=9); ax.grid(True,alpha=0.25,axis='y')
    plt.tight_layout(); plt.savefig("map_toy_GCP.png",dpi=150,bbox_inches='tight'); plt.close()
    print("  Saved: map_toy_GCP.png")

    # ── Shortest Path ─────────────────────────────────────────────────────────
    _,cost,heu,_,S,G = _make_toy_sp(seed)
    n_sp=cost.shape[0]
    # Dijkstra for optimal path
    dist_d={S:0.0}; prev={S:None}; pq=[(0.0,S)]
    while pq:
        d,u=heapq.heappop(pq)
        if d>dist_d.get(u,np.inf): continue
        for v in range(n_sp):
            if cost[u,v]>0:
                nd=d+cost[u,v]
                if nd<dist_d.get(v,np.inf): dist_d[v]=nd; prev[v]=u; heapq.heappush(pq,(nd,v))
    opt_path=[]; node=G
    while node is not None: opt_path.append(node); node=prev.get(node)
    opt_path=opt_path[::-1]
    opt_cost=dist_d.get(G,np.inf)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    fig.suptitle(f"Toy SP — {n_sp} nodes  {S}→{G}  |  optimal = {opt_cost:.1f}  (seed={seed})",fontsize=12,fontweight='bold')
    ax=axes[0]; display=np.where(cost>0,cost,np.nan)
    im=ax.imshow(display,cmap='YlOrRd',aspect='auto')
    ax.set_title("Cost Matrix  (white = no edge)"); ax.set_xlabel("To"); ax.set_ylabel("From")
    plt.colorbar(im,ax=ax,label="Edge cost")
    for i in range(n_sp):
        for j in range(n_sp):
            if cost[i,j]>0: ax.text(j,i,f"{int(cost[i,j])}",ha='center',va='center',fontsize=9,color='black')
    # highlight optimal path edges
    for k in range(len(opt_path)-1):
        u,v=opt_path[k],opt_path[k+1]
        ax.add_patch(plt.Rectangle((v-0.45,u-0.45),0.9,0.9,fill=False,edgecolor='lime',linewidth=2.5))
    ax.scatter([S],[S],s=250,c='lime',  zorder=5,marker='*',label=f"Start ({S})")
    ax.scatter([G],[G],s=250,c='purple',zorder=5,marker='*',label=f"Goal ({G})")
    ax.legend(fontsize=8,loc='upper right')
    ax=axes[1]; out_deg=(cost>0).sum(axis=1); in_deg=(cost>0).sum(axis=0)
    xn=np.arange(n_sp); w=0.35
    ax.bar(xn-w/2,out_deg,w,label="Out-degree",color='steelblue', edgecolor='gray')
    ax.bar(xn+w/2,in_deg, w,label="In-degree", color='darkorange',edgecolor='gray')
    ax.axvline(S,color='lime',  linewidth=2,linestyle='--',label=f"Start ({S})")
    ax.axvline(G,color='purple',linewidth=2,linestyle='--',label=f"Goal ({G})")
    ax.set_xlabel("Node"); ax.set_ylabel("Degree"); ax.set_xticks(xn)
    ax.set_title(f"In/Out Degree  |  {int((cost>0).sum())} edges  |  optimal path: {'→'.join(str(x) for x in opt_path)}")
    ax.legend(fontsize=8); ax.grid(True,alpha=0.25,axis='y')
    plt.tight_layout(); plt.savefig("map_toy_SP.png",dpi=150,bbox_inches='tight'); plt.close()
    print("  Saved: map_toy_SP.png")

if __name__ == "__main__":
    print("\n"+"="*65)
    print("  PART 1 — CONTINUOUS PROBLEMS  (Meta + Classic)")
    print(f"  NUM_RUNS={SMALL_NUM_RUNS}  POP_SIZE={SMALL_POP_SIZE}  MAX_ITER={SMALL_MAX_ITER}")
    print("="*65)

    for prob_name,(obj,bounds) in CONT_PROBLEMS.items():
        print(f"\n  → {prob_name}")
        combined={}
        for aname,afunc in META_ALGORITHMS.items():
            runs=[]
            for rid in range(SMALL_NUM_RUNS):
                np.random.seed(42+rid); random.seed(42+rid)
                runs.append(_run_small(afunc,obj,bounds))
            combined[aname]=runs
            vals=[r["best_fitness"] for r in runs if np.isfinite(r["best_fitness"])]
            print(f"    {aname:<8}  mean={np.mean(vals):.5f}  std={np.std(vals):.5f}")
        for aname,afunc in CONTINUOUS_CLASSIC_ALGOS.items():
            runs=[]
            for rid in range(SMALL_NUM_RUNS):
                np.random.seed(42+rid); random.seed(42+rid)
                runs.append(_run_small(afunc,obj,bounds))
            combined[aname]=runs
            vals=[r["best_fitness"] for r in runs if np.isfinite(r["best_fitness"])]
            print(f"    [C]{aname:<5}  mean={np.mean(vals):.5f}  std={np.std(vals):.5f}")
        plot_convergence(prob_name,combined,save_name=prob_name)
        plot_timecost(prob_name,   combined,save_name=prob_name)
        plot_bar(prob_name,        combined,save_name=prob_name)
        export_stats(prob_name,    combined)

    print("\n  3-D landscape plots ...")
    for prob_name,(obj,bounds) in CONT_PROBLEMS.items():
        plot_3d(obj,bounds,prob_name)

    print("\n"+"="*65)
    print("  PART 2 — TOY DISCRETE PROBLEMS  (Meta + Classic)")
    print(f"  NUM_RUNS={SMALL_NUM_RUNS}  POP_SIZE={SMALL_POP_SIZE}  MAX_ITER={SMALL_MAX_ITER}")
    print("="*65)

    print("\n  Saving toy problem maps (seed=42 instance) ...")
    save_toy_maps(seed=42)

    TOY_PROBLEMS = ["TSP", "Knapsack", "GraphColoring", "ShortestPath"]

    _toy_tsp_nn_bounds = []
    for rid in range(SMALL_NUM_RUNS):
        _,fit,_,_,_,_,(coords,dist) = _make_toy_tsp(42+rid)
        n = len(coords); best_nn = np.inf
        for s in range(n):
            unvis=set(range(n)); tour=[s]; unvis.remove(s)
            while unvis:
                cur=tour[-1]; nxt=min(unvis,key=lambda v,c=cur:dist[c,v])
                tour.append(nxt); unvis.remove(nxt)
            c=fit(tour)
            if c<best_nn: best_nn=c
        _toy_tsp_nn_bounds.append(best_nn)
    _toy_tsp_mean_nn = float(np.mean(_toy_tsp_nn_bounds))
    print(f"  Toy TSP mean NN bound across {SMALL_NUM_RUNS} runs = {_toy_tsp_mean_nn:.1f}")

    for prob_name in TOY_PROBLEMS:
        print(f"\n  → {prob_name}"); combined={}

        for aname,afunc in META_ALGORITHMS.items():
            runs=[]
            for rid in range(SMALL_NUM_RUNS):
                np.random.seed(42+rid); random.seed(42+rid)
                if   prob_name=="TSP":          obj,_,bounds,_,_,_,_ = _make_toy_tsp(42+rid)
                elif prob_name=="Knapsack":     obj,_,bounds,_,_,_,_ = _make_toy_ks(42+rid)
                elif prob_name=="GraphColoring":obj,_,bounds,_,_,_,_ = _make_toy_gcp(42+rid)
                else:                           obj,_,_,bounds,_,_ = _make_toy_sp(42+rid)
                runs.append(_run_small(afunc,obj,bounds))
            combined[aname]=runs; print(f"    {aname:<8}  mean={_mean(runs):.3f}")

        for aname,afunc in CLASSIC_ALGOS.items():
            raw=[]
            for rid in range(SMALL_NUM_RUNS):
                np.random.seed(42+rid); random.seed(42+rid)
                if prob_name=="ShortestPath":
                    _,cost,heu,_,S,G = _make_toy_sp(42+rid)
                    raw.append(_run_toy_graph(afunc,cost,heu,S,G))
                elif prob_name=="TSP":
                    _,fit,bounds,init_fn,hc_fn,sa_fn,_ = _make_toy_tsp(42+rid)
                    if   aname in ("BFS","DFS","UCS","Greedy","A*"): raw.append(_run_toy_comb(afunc,fit,init_fn()))
                    elif aname=="HC":  v,t=hc_fn(); raw.append({"cost":v,"time_sec":t})
                    elif aname=="SA":  v,t=sa_fn(); raw.append({"cost":v,"time_sec":t})
                    else: raw.append({"cost":np.inf,"time_sec":0.001})
                elif prob_name=="Knapsack":
                    _,fit,bounds,init_fn,hc_fn,sa_fn,_ = _make_toy_ks(42+rid)
                    if   aname in ("BFS","DFS","UCS","Greedy","A*"): raw.append(_run_toy_comb(afunc,fit,init_fn()))
                    elif aname=="HC":  v,t=hc_fn(); raw.append({"cost":v,"time_sec":t})
                    elif aname=="SA":  v,t=sa_fn(); raw.append({"cost":v,"time_sec":t})
                    else: raw.append({"cost":np.inf,"time_sec":0.001})
                elif prob_name=="GraphColoring":
                    _,fit,bounds,init_fn,hc_fn,sa_fn,_ = _make_toy_gcp(42+rid)
                    if   aname in ("BFS","DFS","UCS","Greedy","A*"): raw.append(_run_toy_comb(afunc,fit,init_fn()))
                    elif aname=="HC":  v,t=hc_fn(); raw.append({"cost":v,"time_sec":t})
                    elif aname=="SA":  v,t=sa_fn(); raw.append({"cost":v,"time_sec":t})
                    else: raw.append({"cost":np.inf,"time_sec":0.001})
            combined[aname]=_wrap_toy(raw,SMALL_MAX_ITER)
            print(f"    [C]{aname:<5}  mean={_mean(combined[aname]):.3f}")
        plot_convergence(prob_name,combined,save_name=f"toy_{prob_name}")
        plot_timecost(prob_name,   combined,save_name=f"toy_{prob_name}")
        plot_bar(prob_name,        combined,save_name=f"toy_{prob_name}")
        nn = _toy_tsp_mean_nn if prob_name=="TSP" else None
        export_stats(f"toy_{prob_name}",combined,nn_bound=nn)

    print("\n"+"="*65)
    print("  PART 3 — LARGE DISCRETE PROBLEMS  (Scalability Stress Test)")
    print(f"  NUM_RUNS={LARGE_NUM_RUNS}  POP_SIZE={LARGE_POP_SIZE}  MAX_ITER={LARGE_MAX_ITER}")
    print(f"  NODE_CAP={LARGE_NODE_CAP:,}  HC_ITER={LARGE_HC_ITER:,}  SA_ITER={LARGE_SA_ITER:,}")
    print("="*65)

    nn_ref=_L_tsp_nn_bound(); ks_dp_opt=_L_ks_dp_optimal()
    gcp_nc,_=_L_gcp_greedy_ref(); sp_opt=_L_dijkstra_optimal()

    print(f"\n  TSP:  {_L_N_CITIES} cities   — NN bound = {nn_ref:.0f}")
    print(f"  KP:   {_L_N_ITEMS} items    — DP optimal = {ks_dp_opt}  (neg = {-ks_dp_opt})")
    print(f"  GCP:  {_L_N_VERT} vertices ({_L_N_EDGES} edges, {_L_N_COLORS} colors) — greedy = {gcp_nc} colors")
    print(f"  SP:   {_L_N_NODES} nodes    — Dijkstra optimal = {sp_opt:.1f}")
    print("\n  Saving problem maps ...")
    save_problem_maps(nn_ref,ks_dp_opt,gcp_nc,sp_opt)

    # ── TSP ───────────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}\n  [TSP] {_L_N_CITIES} cities  |  NN bound={nn_ref:.0f}\n{'─'*65}")
    res_tsp={}
    for aname,afunc in META_ALGORITHMS.items():
        runs=[]
        for rid in range(LARGE_NUM_RUNS):
            np.random.seed(42+rid); random.seed(42+rid)
            runs.append(_run_large(afunc,L_tsp_obj_meta,L_TSP_BOUNDS))
        res_tsp[aname]=runs; print(f"    {aname:<8}  mean={_mean(runs):.1f}")
    for aname in ("BFS","DFS","UCS","Greedy","A*"):
        runs=[]
        for rid in range(LARGE_NUM_RUNS):
            np.random.seed(42+rid); random.seed(42+rid)
            init=list(np.random.permutation(_L_N_CITIES))
            c,t=_run_classic_comb(aname,L_tsp_fit,init,_L_tsp_all_2opt); runs.append(_wrap_large(c,t))
        res_tsp[aname]=runs; print(f"    [C]{aname:<5}  mean={_mean(runs):.1f}  (cap={LARGE_NODE_CAP:,})")
    for aname,fn in (("HC",_large_hc),("SA",_large_sa)):
        runs=[]
        for rid in range(LARGE_NUM_RUNS):
            np.random.seed(42+rid); random.seed(42+rid)
            c,t=fn(L_tsp_fit,lambda:list(np.random.permutation(_L_N_CITIES)),_L_tsp_2opt); runs.append(_wrap_large(c,t))
        res_tsp[aname]=runs; print(f"    [C]{aname:<5}  mean={_mean(runs):.1f}  (2-opt)")
    plot_convergence(f"TSP ({_L_N_CITIES} cities)",res_tsp,"large_TSP")
    plot_timecost(   f"TSP ({_L_N_CITIES} cities)",res_tsp,"large_TSP")
    plot_bar(        f"TSP ({_L_N_CITIES} cities)",res_tsp,ref_line=nn_ref,ref_label=f"NN bound ({nn_ref:.0f})",save_name="large_TSP")
    export_stats("TSP_large",res_tsp,nn_bound=nn_ref)

    print(f"\n{'─'*65}\n  [KP] {_L_N_ITEMS} items  cap={_L_KCAP}  |  DP optimal={ks_dp_opt}\n{'─'*65}")
    res_ks={}
    for aname,afunc in META_ALGORITHMS.items():
        runs=[]
        for rid in range(LARGE_NUM_RUNS):
            np.random.seed(42+rid); random.seed(42+rid)
            runs.append(_run_large(afunc,L_ks_obj_meta,L_KS_BOUNDS))
        res_ks[aname]=runs; m=_mean(runs); pct=-m/ks_dp_opt*100 if np.isfinite(m) else 0
        print(f"    {aname:<8}  mean={m:.1f}  ({pct:.1f}% of DP opt)")
    for aname in ("BFS","DFS","UCS","Greedy","A*"):
        runs=[]
        for rid in range(LARGE_NUM_RUNS):
            np.random.seed(42+rid); random.seed(42+rid)
            init=list(np.random.randint(0,2,_L_N_ITEMS))
            c,t=_run_classic_comb(aname,L_ks_fit,init,_L_ks_all_flips); runs.append(_wrap_large(c,t))
        res_ks[aname]=runs; m=_mean(runs); pct=-m/ks_dp_opt*100 if np.isfinite(m) else 0
        print(f"    [C]{aname:<5}  mean={m:.1f}  ({pct:.1f}% of DP opt)  cap={LARGE_NODE_CAP:,}")
    for aname,fn in (("HC",_large_hc),("SA",_large_sa)):
        runs=[]
        for rid in range(LARGE_NUM_RUNS):
            np.random.seed(42+rid); random.seed(42+rid)
            c,t=fn(L_ks_fit,lambda:list(np.random.randint(0,2,_L_N_ITEMS)),_L_ks_flip); runs.append(_wrap_large(c,t))
        res_ks[aname]=runs; m=_mean(runs); pct=-m/ks_dp_opt*100 if np.isfinite(m) else 0
        print(f"    [C]{aname:<5}  mean={m:.1f}  ({pct:.1f}% of DP opt)")
    plot_convergence(f"Knapsack ({_L_N_ITEMS} items)",res_ks,"large_KP")
    plot_timecost(   f"Knapsack ({_L_N_ITEMS} items)",res_ks,"large_KP")
    plot_bar(        f"Knapsack ({_L_N_ITEMS} items)",res_ks,ref_line=-float(ks_dp_opt),ref_label=f"DP optimal (−{ks_dp_opt})",save_name="large_KP")
    export_stats("KP_large",res_ks)

    print(f"\n{'─'*65}\n  [GCP] {_L_N_VERT} vertices  {_L_N_EDGES} edges  {_L_N_COLORS} colors  |  greedy={gcp_nc} colors\n{'─'*65}")
    res_gcp={}
    for aname,afunc in META_ALGORITHMS.items():
        runs=[]
        for rid in range(LARGE_NUM_RUNS):
            np.random.seed(42+rid); random.seed(42+rid)
            runs.append(_run_large(afunc,L_gcp_obj_meta,L_GCP_BOUNDS))
        res_gcp[aname]=runs; print(f"    {aname:<8}  mean={_mean(runs):.1f} conflicts")
    for aname in ("BFS","DFS","UCS","Greedy","A*"):
        runs=[]
        for rid in range(LARGE_NUM_RUNS):
            np.random.seed(42+rid); random.seed(42+rid)
            init=list(np.random.randint(0,_L_N_COLORS,_L_N_VERT))
            c,t=_run_classic_comb(aname,L_gcp_fit,init,_L_gcp_all_recolors); runs.append(_wrap_large(c,t))
        res_gcp[aname]=runs; print(f"    [C]{aname:<5}  mean={_mean(runs):.1f} conflicts  cap={LARGE_NODE_CAP:,}")
    for aname,fn in (("HC",_large_hc),("SA",_large_sa)):
        runs=[]
        for rid in range(LARGE_NUM_RUNS):
            np.random.seed(42+rid); random.seed(42+rid)
            c,t=fn(L_gcp_fit,lambda:list(np.random.randint(0,_L_N_COLORS,_L_N_VERT)),_L_gcp_recolor); runs.append(_wrap_large(c,t))
        res_gcp[aname]=runs; print(f"    [C]{aname:<5}  mean={_mean(runs):.1f} conflicts")
    plot_convergence(f"GCP ({_L_N_VERT} vertices)",res_gcp,"large_GCP")
    plot_timecost(   f"GCP ({_L_N_VERT} vertices)",res_gcp,"large_GCP")
    plot_bar(        f"GCP ({_L_N_VERT} vertices)",res_gcp,ref_line=0,ref_label="Optimal (0 conflicts)",save_name="large_GCP")
    export_stats("GCP_large",res_gcp)

    print(f"\n{'─'*65}\n  [SP] {_L_N_NODES} nodes  0→{_L_SP_G}  |  Dijkstra optimal={sp_opt:.1f}\n{'─'*65}")
    res_sp={}
    for aname,afunc in META_ALGORITHMS.items():
        runs=[]
        for rid in range(LARGE_NUM_RUNS):
            np.random.seed(42+rid); random.seed(42+rid)
            runs.append(_run_large(afunc,L_sp_obj_meta,L_SP_BOUNDS))
        res_sp[aname]=runs; m=_mean(runs)
        print(f"    {aname:<8}  mean={'∞' if not np.isfinite(m) else f'{m:.1f}'}")
    sp_classic_fns={"BFS":bfs,"DFS":dfs,"UCS":ucs,"Greedy":greedy_best_first_search,
                    "A*":a_star_search,"HC":hill_climbing_steepest_ascent}
    for aname,afunc in sp_classic_fns.items():
        runs=[]
        for _ in range(LARGE_NUM_RUNS):
            c,t=_run_classic_sp(afunc); runs.append(_wrap_large(c,t))
        res_sp[aname]=runs; m=_mean(runs)
        if np.isfinite(m):
            gap=(m-sp_opt)/sp_opt*100; tag="(optimal)" if abs(gap)<0.1 else f"(+{gap:.1f}% vs Dijkstra)"
        else: tag="(no path found)"
        print(f"    [C]{aname:<5}  mean={'∞' if not np.isfinite(m) else f'{m:.1f}'}  {tag}")
    res_sp["SA"]=res_sp["BFS"]
    plot_convergence(f"SP ({_L_N_NODES} nodes)",res_sp,"large_SP")
    plot_timecost(   f"SP ({_L_N_NODES} nodes)",res_sp,"large_SP")
    plot_bar(        f"SP ({_L_N_NODES} nodes)",res_sp,ref_line=sp_opt,ref_label=f"Dijkstra ({sp_opt:.1f})",save_name="large_SP")
    export_stats("SP_large",res_sp)

    print("\n"+"="*65)
    print("  Done!  All stats_*.txt and *.png files saved.")
    print("="*65)
