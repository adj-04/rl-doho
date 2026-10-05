from common3 import *
import load2
X = load_all(); L = load_long(); SP = load_splits()
XM = X[X.method.isin(METHODS)]
S = XM.groupby(["dataset", "method"]).agg(fit=("fit", "mean"), acc=("test_acc", "mean"), nf=("n_feats", "mean")).reset_index()
# --- tabular per-dataset table (same layout as genes.tex)
Lns = []
for m in METHODS:
    c = [r"\textbf{RL-DOHO}" if m == "RL-DOHO" else m]
    for ds in TAB:
        Sd = S[S.dataset == ds].set_index("method"); r = Sd.loc[m]
        c += [bold(f"{r.fit:.3f}", r.fit == Sd.fit.min()), bold(f"{100*r.acc:.1f}", r.acc == Sd.acc.max()), bold(r0(r.nf), r.nf == Sd.nf.min())]
    Lns.append(" & ".join(c) + r" \\")
    if m in ("DOHO", "BGWO"): Lns.append(r"\midrule")
open(T + "tab.tex", "w").write("\n".join(Lns))
# --- pooled over 110 runs
PR = pd.read_csv("pooled_all11.csv"); Lns = []
for c in METHODS[1:]:
    cells = [c]
    for met in ["fit", "test_acc", "test_f1"]:
        r = PR[(PR.metric == met) & (PR.vs == c)].iloc[0]; cells += [f"{r.W}/{r['T']}/{r.L}", bold(pf(r.holm), r.holm < 0.05)]
    Lns.append(" & ".join(cells) + r" \\")
    if c in ("DOHO", "BGWO"): Lns.append(r"\midrule")
open(T + "pooled.tex", "w").write("\n".join(Lns))
# --- family over 11 datasets
Fm = X[X.method.isin(FAMILY)].copy(); Fm["r"] = Fm.groupby(["dataset", "seed"]).fit.transform(lambda x: rankdata(np.round(x, 12)))
Lns = []
for ds in DS:
    if ds == "Colon" or ds == "WDBC": Lns.append(r"\midrule")
    P2 = Fm[Fm.dataset == ds].pivot(index="seed", columns="method", values="fit")[FAMILY]; p = friedmanchisquare(*[P2[c] for c in FAMILY]).pvalue
    rr = Fm[Fm.dataset == ds].groupby("method").r.mean()[FAMILY]
    Lns.append(" & ".join([ds] + [bold(f"{rr[a]:.2f}", rr[a] == rr.min()) for a in FAMILY] + [pf(p)]) + r" \\")
rr = Fm.groupby("method").r.mean()[FAMILY]; n = Fm.groupby(["dataset", "seed"]).ngroups; Lns.append(r"\midrule")
Lns.append(" & ".join([f"Mean ({n} runs)"] + [bold(f"{rr[a]:.2f}", round(rr[a], 2) == round(rr.min(), 2)) for a in FAMILY] + ["--"]) + r" \\")
Fm["f12"] = Fm.fit.round(12); Fm["rmin"] = Fm.groupby(["dataset", "seed"]).f12.rank(method="min"); Fm["rmax"] = Fm.groupby(["dataset", "seed"]).f12.rank(method="max")
first = (Fm[Fm.rmin == 1].groupby("method").size() / n * 100).reindex(FAMILY).fillna(0); last = (Fm[Fm.rmax == 4].groupby("method").size() / n * 100).reindex(FAMILY).fillna(0)
Lns.append(" & ".join(["Best or tied (\\%)"] + [bold(r0(first[a]), r0(first[a]) == r0(first.max())) for a in FAMILY] + ["--"]) + r" \\")
Lns.append(" & ".join(["Worst or tied (\\%)"] + [bold(r0(last[a]), r0(last[a]) == r0(last.min())) for a in FAMILY] + ["--"]) + r" \\")
open(T + "family.tex", "w").write("\n".join(Lns)); print("family", rr.round(2).to_dict(), first.round(1).to_dict(), last.round(1).to_dict())
# --- ablation (same budget), three groups
A2 = load2.load_abl("v2_ablation"); A2["group"] = np.where(A2.dataset == "Rheumatic", "clinical", "gene")
ren = {"RL-DOHO v2 (UCB, gain reward)": "RL-DOHO", "UCB, old reward": "UCB, old reward", "No tie-break": "No tie-break", "Dense start (as Notebooks 1-5)": "Dense start"}
A2 = A2[A2.variant.isin(["UCB, old reward", "No tie-break", "Dense start (as Notebooks 1-5)"])].rename(columns={"variant": "method"}); A2["method"] = A2.method.map(ren)
VMAP = {"RL-DOHO": "RL-DOHO (UCB1, gain reward)", "RL-DOHO [Q-learning]": "Q-learning controller", "RL-DOHO [random arm]": "Random arm",
        "RL-DOHO [PERTURB only]": "PERTURB only", "UCB, old reward": "UCB1, old reward", "No tie-break": "No tie-break", "Dense start": "Dense start",
        "RL-DOHO [no intervention]": "No intervention (DO)"}
AB = pd.concat([X[X.method.isin(VMAP)][["dataset", "method", "seed", "fit", "test_acc", "n_feats", "group"]], A2[["dataset", "method", "seed", "fit", "test_acc", "n_feats", "group"]]])
Lns = []
for v, lab in VMAP.items():
    cells = [lab]
    for g in ["clinical", "gene", "tabular"]:
        G = AB[AB.group == g]
        if v not in set(G.method): cells += ["--", "--"]; continue
        cells.append(f"{G[G.method == v].fit.mean():.4f}")
        if v == "RL-DOHO": cells.append("--")
        else:
            w, t, l, p, n_, _ = pair(G, "RL-DOHO", v); cells.append(f"{w}/{t}/{l} ({pf(p)})")
    if v == "RL-DOHO": cells.append("--")
    elif all(v in set(AB[AB.group == g].method) for g in ["clinical", "gene", "tabular"]):
        w, t, l, p, n_, _ = pair(AB, "RL-DOHO", v); cells.append(f"{w}/{t}/{l} ({pf(p)})")
    else: cells.append("--")
    Lns.append(" & ".join(cells) + r" \\")
open(T + "ablation.tex", "w").write("\n".join(Lns)); print("\n".join(Lns))
# --- long budget: three groups + all
L2 = load2.load_abl("v2_long"); L2 = L2[L2.variant == "UCB, old reward"].rename(columns={"variant": "method"}); L2["group"] = np.where(L2.dataset == "Rheumatic", "clinical", "gene")
LL = pd.concat([L, L2]); LMAP = {"UCB1 (3x)": "UCB1, gain reward", "Q-learning (3x)": "Q-learning", "Random arm (3x)": "Random arm", "UCB, old reward": "UCB1, old reward"}
Lns = []
for v, lab in LMAP.items():
    cells = [lab]
    for g in ["clinical", "gene", "tabular"]:
        G = LL[LL.group == g]
        if v not in set(G.method): cells += ["--", "--"]; continue
        cells.append(f"{G[G.method == v].fit.mean():.4f}")
        cells.append("--" if v == "UCB1 (3x)" else "{}/{}/{} ({})".format(*pair(G, "UCB1 (3x)", v)[:3], pf(pair(G, "UCB1 (3x)", v)[3])))
    if v in ("UCB1 (3x)", "UCB, old reward"): cells.append("--")
    else: w, t, l, p, n_, _ = pair(L, "UCB1 (3x)", v); cells.append(f"{w}/{t}/{l} ({pf(p)})")
    Lns.append(" & ".join(cells) + r" \\")
open(T + "long.tex", "w").write("\n".join(Lns)); print("\n".join(Lns))
# --- add-on table
Lns = []
names = ["RL-DOHO", "GA", "GA+SI", "HO", "HO+SI"]
for m in names:
    Lns.append(" & ".join([m] + [f"{X[(X.method == m) & (X.group == g)].fit.mean():.4f}" for g in ["clinical", "gene", "tabular"]] + [f"{X[X.method == m].n_feats.mean():.1f}"]) + r" \\")
open(T + "addon_means.tex", "w").write("\n".join(Lns))
Lns = []; h6 = holm([pair(X, "GA+SI", "GA")[3], pair(X, "HO+SI", "HO")[3]])
for (a, b), lab in [(("GA+SI", "GA"), "GA+SI vs.\\ GA"), (("HO+SI", "HO"), "HO+SI vs.\\ HO"), (("RL-DOHO", "GA+SI"), "RL-DOHO vs.\\ GA+SI"), (("RL-DOHO", "HO+SI"), "RL-DOHO vs.\\ HO+SI")]:
    cells = [lab] + ["{}/{}/{}".format(*pair(X[X.group == g], a, b)[:3]) for g in ["clinical", "gene", "tabular"]]
    w, t, l, p, n_, _ = pair(X, a, b)
    pp = h6[0] if b == "GA" and a == "GA+SI" else h6[1] if (a, b) == ("HO+SI", "HO") else p
    cells += [f"{w}/{t}/{l}", bold(pf(pp), pp < 0.05)]
    wt, tt, lt, ptt, _, _ = pair(X, a, b, "test_acc", True); cells += [f"{wt}/{tt}/{lt}", bold(pf(ptt), ptt < 0.05)]
    Lns.append(" & ".join(cells) + r" \\")
    if lab.startswith("HO+SI"): Lns.append(r"\midrule")
open(T + "addon.tex", "w").write("\n".join(Lns)); print("\n".join(Lns))
# --- kappa table
KAP = {"RL-DOHO [κ=1]": 1, "RL-DOHO": 2, "RL-DOHO [κ=3]": 3, "RL-DOHO [κ=4]": 4}
K = X[X.method.isin(KAP)].copy(); K["kappa"] = K.method.map(KAP); K["r"] = K.groupby(["dataset", "seed"]).fit.transform(lambda x: rankdata(np.round(x, 12)))
Lns = []
for m, k in KAP.items():
    cells = [f"$\\kappa={k}$" + (" (default)" if k == 2 else "")] + [f"{K[(K.kappa == k) & (K.group == g)].fit.mean():.4f}" for g in ["clinical", "gene", "tabular"]]
    rk = K.groupby("kappa").r.mean(); cells.append(bold(f"{rk[k]:.2f}", rk[k] == rk.min()))
    if k == 2: cells += ["--", "--"]
    else:
        w, t, l, p, _, _ = pair(X, m, "RL-DOHO"); cells += [f"{w}/{t}/{l}", bold(pf(p), p < 0.05)]
    Lns.append(" & ".join(cells) + r" \\")
open(T + "kappa.tex", "w").write("\n".join(Lns)); print("\n".join(Lns))
# --- splits table
MB = METHODS[:7] + ADDONS + METHODS[7:]
SPL = ["split 42 (original)", "split 1", "split 2"]
SP["r"] = SP.groupby(["dataset", "seed"]).fit.transform(lambda x: rankdata(np.round(x, 12))); SP["ra"] = SP.groupby(["dataset", "seed"]).test_acc.transform(lambda x: rankdata(-np.round(x, 12)))
f = SP.pivot_table(index="method", columns="dataset", values="fit"); a = SP.pivot_table(index="method", columns="dataset", values="test_acc")
rk = SP.groupby("method").r.mean(); ra = SP.groupby("method").ra.mean()
Lns = []
for m in MB:
    cells = [r"\textbf{RL-DOHO}" if m == "RL-DOHO" else m]
    cells += [bold(f"{f.loc[m, s]:.4f}", f.loc[m, s] == f[s].min()) for s in SPL]
    cells += [bold(f"{100*a.loc[m, s]:.2f}", a.loc[m, s] == a[s].max()) for s in SPL]
    cells += [bold(f"{rk[m]:.2f}", rk[m] == rk.min()), bold(f"{ra[m]:.2f}", ra[m] == ra.min())]
    if m == "RL-DOHO": cells.append("--")
    else: w, t, l, p, _, _ = pair(SP, "RL-DOHO", m); cells.append(f"{w}/{t}/{l}")
    Lns.append(" & ".join(cells) + r" \\")
    if m in ("DOHO", "BGWO", "HO+SI"): Lns.append(r"\midrule")
open(T + "splits.tex", "w").write("\n".join(Lns))
# --- cost table
TK = pd.read_csv(R3 + "rl_doho_results_v3_knn/v3_timing.csv"); TK["dataset"] = TK.dataset.map(DSN)
TR = pd.read_csv(R3 + "rl_doho_results_v3_rheum/v3_timing.csv")
OB = pd.read_csv(R3 + "rl_doho_results_v3_knn/v3_overhead_benchmark.csv", index_col=0)
WR = WRAP + ADDONS
nt = TK[TK.dataset.isin(TAB)].groupby("method").n_unique.mean(); nr = TR.groupby("method").n_unique.mean()
Lns = []
for m in WR:
    cells = [m, r0(nr[m]), r0(nt[m])] + [f"{OB.loc[m, c]:.2f}" for c in ["14", "44", "2000", "7129"]]
    Lns.append(" & ".join(cells) + r" \\")
    if m == "DOHO": Lns.append(r"\midrule")
open(T + "cost.tex", "w").write("\n".join(Lns)); print("\n".join(Lns))
