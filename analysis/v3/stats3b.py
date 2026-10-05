from common3 import *
import pickle
SP = load_splits()
MB = METHODS[:7] + ADDONS + METHODS[7:]
print(SP.groupby(["dataset"]).size())
SP["r"] = SP.groupby(["dataset", "seed"]).fit.transform(lambda x: rankdata(x)); SP["ra"] = SP.groupby(["dataset", "seed"]).test_acc.transform(lambda x: rankdata(-x))
print(SP.pivot_table(index="method", columns="dataset", values="fit").reindex(MB).round(4))
print(SP.pivot_table(index="method", columns="dataset", values="test_acc").reindex(MB).round(4))
print(SP.groupby("method")[["r", "ra", "fit", "test_acc", "n_feats"]].mean().reindex(MB).round(3))
print(SP.pivot_table(index="method", columns="dataset", values="r").reindex(MB).round(2))
rows = []
for m, hb in [("fit", False), ("test_acc", True)]:
    tmp = [(m, c) + pair(SP, "RL-DOHO", c, m, hb)[:4] for c in MB if c != "RL-DOHO"]
    for x, h in zip(tmp, holm([x[-1] for x in tmp])): rows.append(x + (h,))
PB = pd.DataFrame(rows, columns=["metric", "vs", "W", "T", "L", "p", "holm"]); print(PB.round(4).to_string(index=False)); PB.to_csv("splits_pooled.csv", index=False)
Fm = SP[SP.method.isin(FAMILY)].copy(); Fm["rf"] = Fm.groupby(["dataset", "seed"]).fit.transform(lambda x: rankdata(x))
print(Fm.pivot_table(index="method", columns="dataset", values="rf").reindex(FAMILY).assign(ALL=Fm.groupby("method").rf.mean()).round(2))
P = SP.pivot_table(index=["dataset", "seed"], columns="method", values="fit")[MB]
print("Friedman fit 15 blocks:", friedmanchisquare(*[P[c] for c in MB]).pvalue)
P = SP.pivot_table(index=["dataset", "seed"], columns="method", values="test_acc")[MB]
print("Friedman acc:", friedmanchisquare(*[P[c] for c in MB]).pvalue)
# best fitness per split and which runs hit it
for ds, g in SP.groupby("dataset"):
    b = g.fit.min(); print(ds, "best", round(b, 5), g[np.isclose(g.fit, b, atol=1e-9)].groupby("method").size().to_dict())
# ---------- cost
TK = pd.read_csv(R3 + "rl_doho_results_v3_knn/v3_timing.csv"); TR = pd.read_csv(R3 + "rl_doho_results_v3_rheum/v3_timing.csv")
ck = pickle.load(open(R3 + "rl_doho_results_v3_knn/v3_eval_cost.pkl", "rb")); cr = pickle.load(open(R3 + "rl_doho_results_v3_rheum/v3_eval_cost_splits1_2.pkl", "rb"))
print("knn eval cost:", {k: (v[0], round(v[1] / max(v[0], 1), 4)) for k, v in ck.items()}); print("rheum eval cost:", {k: (v[0], round(v[1] / max(v[0], 1), 3)) for k, v in cr.items()})
TK["dataset"] = TK.dataset.map(DSN)
WR = WRAP + ADDONS + ["RL-DOHO [Q-learning]"]
tabT = TK[TK.dataset.isin(TAB)]
print("tabular n_unique per run:\n", tabT.groupby("method").n_unique.mean().reindex(WR).round(1))
print("tabular overhead s:\n", tabT.groupby("method").overhead_sec.mean().reindex(WR).round(3))
print("gene n_unique (addons, Q):\n", TK[TK.dataset.isin(GENE)].groupby("method").n_unique.mean().round(1))
print("rheum splits n_unique:\n", TR.groupby("method").n_unique.mean().reindex(WR).round(1), "\n", TR.groupby(["method", "dataset"]).n_unique.mean().unstack().round(1))
OB = pd.read_csv(R3 + "rl_doho_results_v3_knn/v3_overhead_benchmark.csv", index_col=0); print(OB.round(3))
