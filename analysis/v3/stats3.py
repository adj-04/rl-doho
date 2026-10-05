from common3 import *
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
X = load_all(); L = load_long(); SP = load_splits()
out = {}
# ---------- tabular per-dataset (11 methods)
XT = X[X.group == "tabular"]
print("=== tabular: per-dataset Holm-Wilcoxon (RL-DOHO vs each), fitness & test")
for ds in TAB:
    for m, hb in [("fit", False), ("test_acc", True), ("test_f1", True)]:
        P = XT[XT.dataset == ds].pivot(index="seed", columns="method", values=m)[METHODS]
        tmp = []
        for c in METHODS[1:]:
            w, t, l = wtl(P["RL-DOHO"], P[c], hb)
            try: p = wilcoxon(P["RL-DOHO"], P[c], zero_method="zsplit").pvalue
            except ValueError: p = 1.0
            tmp.append((c, w, t, l, p))
        h = holm([x[-1] for x in tmp]); fp = friedmanchisquare(*[P[c] for c in METHODS]).pvalue
        sig = [(c, w, t, l, round(hh, 4)) for (c, w, t, l, p), hh in zip(tmp, h) if hh < 0.05]
        print(ds, m, "friedman", f"{fp:.2g}", "sig:", sig)
# ---------- pooled over scopes
def pooled(Xs, methods):
    rows = []
    for m, hb in [("fit", False), ("test_acc", True), ("test_f1", True)]:
        tmp = [(m, c) + pair(Xs, "RL-DOHO", c, m, hb)[:4] for c in methods]
        for x, h in zip(tmp, holm([x[-1] for x in tmp])): rows.append(x + (h,))
    return pd.DataFrame(rows, columns=["metric", "vs", "W", "T", "L", "p", "holm"])
XM = X[X.method.isin(METHODS)]
for scope, dss in [("all11", DS), ("six", DS[:6]), ("tab", TAB)]:
    P = pooled(XM[XM.dataset.isin(dss)], METHODS[1:]); P.to_csv(f"pooled_{scope}.csv", index=False)
    print(f"\n=== pooled {scope}\n", P.round(4).to_string(index=False))
# ---------- dataset-level ranks + CD (11 datasets)
S = XM.groupby(["dataset", "method"]).agg(fit=("fit", "mean"), acc=("test_acc", "mean"), nf=("n_feats", "mean"), f1=("test_f1", "mean")).reset_index()
S.to_csv("summary3.csv", index=False)
for m, hb in [("fit", False), ("acc", True)]:
    P = S.pivot(index="dataset", columns="method", values=m).loc[DS, METHODS]
    Rk = P.apply(lambda r: pd.Series(rankdata(-r.values if hb else r.values), index=r.index), axis=1)
    st, p = friedmanchisquare(*[P[c] for c in METHODS])
    print(f"\ndataset-level {m} (11 ds): chi2={st:.2f} p={p:.3g}", Rk.mean().sort_values().round(2).to_dict())
    k, N = len(METHODS), len(DS); cd = 3.219 * np.sqrt(k * (k + 1) / (6 * N)); print(" CD =", round(cd, 2))
    mr = Rk.mean(); pairs = [(a, b) for i, a in enumerate(METHODS) for b in METHODS[i + 1:] if abs(mr[a] - mr[b]) > cd]
    print(" pairs beyond CD:", pairs)
    Rk.to_csv(f"dsranks_{m}.csv")
# ---------- per-run rank among all 11 methods, by group
XM2 = XM.copy(); XM2["r"] = XM2.groupby(["dataset", "seed"]).fit.transform(lambda x: rankdata(x))
XM2["ra"] = XM2.groupby(["dataset", "seed"]).test_acc.transform(lambda x: rankdata(-x))
print("\nper-run rank fitness by group:\n", XM2.pivot_table(index="method", columns="group", values="r").reindex(METHODS).assign(ALL=XM2.groupby("method").r.mean()).round(2))
print("\nper-run rank test acc by group:\n", XM2.pivot_table(index="method", columns="group", values="ra").reindex(METHODS).assign(ALL=XM2.groupby("method").ra.mean()).round(2))
# ---------- family over 11 datasets
Fm = X[X.method.isin(FAMILY)].copy()
Fm["r"] = Fm.groupby(["dataset", "seed"]).fit.transform(lambda x: rankdata(x))
Fm["rmin"] = Fm.groupby(["dataset", "seed"]).fit.rank(method="min"); Fm["rmax"] = Fm.groupby(["dataset", "seed"]).fit.rank(method="max")
n = Fm.groupby(["dataset", "seed"]).ngroups
print("\nfamily rank by group:\n", Fm.pivot_table(index="method", columns="group", values="r").reindex(FAMILY).assign(ALL=Fm.groupby("method").r.mean()).round(2))
print(" best-or-tied %:", (Fm[Fm.rmin == 1].groupby("method").size() / n * 100).reindex(FAMILY).round(0).to_dict(),
      " worst-or-tied %:", (Fm[Fm.rmax == 4].groupby("method").size() / n * 100).reindex(FAMILY).fillna(0).round(0).to_dict())
for grp in ["clinical", "gene", "tabular"]:
    G = Fm[Fm.group == grp]; ng = G.groupby(["dataset", "seed"]).ngroups
    print(f"  {grp}: worst-or-tied %", (G[G.rmax == 4].groupby("method").size() / ng * 100).reindex(FAMILY).fillna(0).round(0).to_dict())
P = Fm.pivot_table(index=["dataset", "seed"], columns="method", values="fit"); gap = P.sub(P.min(axis=1), axis=0)
print(" gap to family best mean:", gap.mean().reindex(FAMILY).round(4).to_dict(), "max:", gap.max().reindex(FAMILY).round(4).to_dict())
for a in ["HO", "DOHO", "DO"]: print("  RL-DOHO vs", a, "110 runs:", pair(X, "RL-DOHO", a)[:5])
# ---------- H1-H3 replication on tabular, H5-H8
print("\n=== hypotheses")
print("H1 tab", pair(XT, "RL-DOHO", "DO")[:5], " H1 all11", pair(X, "RL-DOHO", "DO")[:5])
print("H2 tab", pair(XT, "RL-DOHO", "HO")[:5], " H2 all11", pair(X, "RL-DOHO", "HO")[:5])
for grp in ["clinical", "gene", "tabular", None]:
    Lg = L if grp is None else L[L.group == grp]
    print(f"3x [{grp}] UCB vs random", pair(Lg, "UCB1 (3x)", "Random arm (3x)")[:5], " Q vs random", pair(Lg, "Q-learning (3x)", "Random arm (3x)")[:5],
          " Q vs UCB", pair(Lg, "Q-learning (3x)", "UCB1 (3x)")[:5])
print("3x means by group:\n", L.pivot_table(index="method", columns="group", values="fit").round(4))
h6 = [pair(X, "GA+SI", "GA"), pair(X, "HO+SI", "HO")]; print("H6", [x[:5] for x in h6], "holm", holm([x[3] for x in h6]))
for grp in ["clinical", "gene", "tabular"]:
    G = X[X.group == grp]
    print(f"  [{grp}] GA+SI vs GA", pair(G, "GA+SI", "GA")[:5], " HO+SI vs HO", pair(G, "HO+SI", "HO")[:5],
          " RL vs GA+SI", pair(G, "RL-DOHO", "GA+SI")[:5], " RL vs HO+SI", pair(G, "RL-DOHO", "HO+SI")[:5])
for a, b in [("GA+SI", "GA"), ("HO+SI", "HO")]:
    print(f"  test {a} vs {b}", pair(X, a, b, "test_acc", True)[:5], "n_feats mean", X[X.method == a].n_feats.mean().round(1), X[X.method == b].n_feats.mean().round(1))
print("  RL vs GA+SI all", pair(X, "RL-DOHO", "GA+SI")[:5], " RL vs HO+SI all", pair(X, "RL-DOHO", "HO+SI")[:5], " GA+SI vs HO+SI", pair(X, "GA+SI", "HO+SI")[:5])
# per-dataset Wilcoxon HO+SI vs HO
for ds in DS:
    P = X[X.dataset == ds].pivot(index="seed", columns="method", values="fit")
    try: p = wilcoxon(P["HO+SI"], P["HO"], zero_method="zsplit").pvalue
    except ValueError: p = 1
    print(f"   {ds:12s} HO+SI vs HO", wtl(P["HO+SI"], P["HO"]), round(p, 4), " mean", round(P['HO+SI'].mean(), 4), round(P['HO'].mean(), 4))
KAP = {"RL-DOHO [κ=1]": 1, "RL-DOHO": 2, "RL-DOHO [κ=3]": 3, "RL-DOHO [κ=4]": 4}
K = X[X.method.isin(KAP)].copy(); K["kappa"] = K.method.map(KAP)
P = K.pivot_table(index=["dataset", "seed"], columns="kappa", values="fit").dropna()
print("H7 Friedman 110:", friedmanchisquare(*[P[c] for c in P]), P.rank(axis=1).mean().round(2).to_dict())
for grp, dss in [("clinical", ["Rheumatic"]), ("gene", GENE), ("tabular", TAB)]:
    Pg = P.loc[P.index.get_level_values(0).isin(dss)]
    print(f"  [{grp}] friedman p={friedmanchisquare(*[Pg[c] for c in Pg]).pvalue:.4f} ranks", Pg.rank(axis=1).mean().round(2).to_dict(), "means", Pg.mean().round(4).to_dict())
for k in (1, 3, 4): print(f"  kappa {k} vs 2:", pair(X, f"RL-DOHO [κ={k}]", "RL-DOHO")[:5], " test:", pair(X, f"RL-DOHO [κ={k}]", "RL-DOHO", "test_acc", True)[:5])
# controllers same budget
CT = ["RL-DOHO", "RL-DOHO [Q-learning]", "RL-DOHO [random arm]", "RL-DOHO [PERTURB only]", "RL-DOHO [no intervention]"]
print("\nsame-budget controllers mean fit by group:\n", X[X.method.isin(CT)].pivot_table(index="method", columns="group", values="fit").reindex(CT).round(4))
for v in CT[1:]:
    print(f"  RL-DOHO vs {v}: all", pair(X, "RL-DOHO", v)[:5], *[(g, pair(X[X.group == g], "RL-DOHO", v)[:3]) for g in ["clinical", "gene", "tabular"]])
print("  Q vs random same budget all:", pair(X, "RL-DOHO [Q-learning]", "RL-DOHO [random arm]")[:5])
