import sys; sys.path.insert(0, "/home/claude/paper/v3final/an")
from load3 import *
from scipy.stats import friedmanchisquare, wilcoxon, binomtest, rankdata
TIE = 1e-12
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, j in enumerate(o): run = max(run, (m - i) * ps[j]); adj[j] = min(1, run)
    return adj
def wtl(a, b, hb=False):
    d = np.asarray(a) - np.asarray(b); w = int((d > TIE).sum() if hb else (d < -TIE).sum()); t = int((np.abs(d) <= TIE).sum()); return w, t, len(d) - w - t
def pair(X, a, b, metric="fit", hb=False, col="method"):
    P = X.pivot_table(index=["dataset", "seed"], columns=col, values=metric)
    x, y = P[a].dropna().align(P[b].dropna(), join="inner"); w, t, l = wtl(x, y, hb)
    p = binomtest(w, w + l).pvalue if w + l else 1.0
    return w, t, l, p, len(x), float((x - y).mean())
def pf(p):
    if p < 0.001: return r"$<$0.001"
    t = f"{p:.3f}"
    return f"{p:.4f}" if (p < 0.05 and float(t) >= 0.05) else t
def r0(x): return str(int(np.floor(x + 0.5)))
def bold(s, c): return r"\textbf{" + s + "}" if c else s
T = "/home/claude/paper/v3final/tex/tables/"
