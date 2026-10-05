from common3 import *
S = pd.read_csv("/home/claude/paper/v3final/res4/rl_doho_results_v4_svm/v4_all_runs.csv")
K = load_all(); K = K[K.group == "tabular"]
P = "RL-DOHO [PERTURB only]"
comp = [("RL-DOHO vs.\\ DO", "RL-DOHO", "DO"), ("RL-DOHO vs.\\ HO", "RL-DOHO", "HO"), ("RL-DOHO vs.\\ DOHO", "RL-DOHO", "DOHO"),
        ("GA vs.\\ RL-DOHO", "GA", "RL-DOHO"), ("RL-DOHO vs.\\ BPSO", "RL-DOHO", "BPSO"), ("RL-DOHO vs.\\ BGWO", "RL-DOHO", "BGWO"),
        ("RL-DOHO vs.\\ LASSO", "RL-DOHO", "LASSO"), ("HO+SI vs.\\ HO", "HO+SI", "HO"), ("GA+SI vs.\\ GA", "GA+SI", "GA"),
        ("PERTURB only vs.\\ RL-DOHO", P, "RL-DOHO")]
out = []
for lab, a, b in comp:
    c = [lab]
    for D in (S, K):
        w, t, l, p, n, _ = pair(D, a, b); c += [f"{w}/{t}/{l}", bold(pf(p), p < 0.05)]
    out.append(" & ".join(c) + r" \\"); print(out[-1])
    if lab.startswith("RL-DOHO vs.\\ LASSO"): out.append(r"\midrule")
open(T + "svm.tex", "w").write("\n".join(out))
# ranks + spearman
MS = METHODS
def ranks(D):
    D = D[D.method.isin(MS)].copy(); D["r"] = D.groupby(["dataset", "seed"]).fit.transform(lambda x: rankdata(np.round(x, 12))); return D.groupby("method").r.mean()
rs, rk = ranks(S), ranks(K)
from scipy.stats import spearmanr
print("ranks SVM", rs.sort_values().round(2).to_dict()); print("ranks KNN", rk.sort_values().round(2).to_dict()); print("spearman", spearmanr(rs, rk.reindex(rs.index)).correlation)
F = S[S.method.isin(FAMILY)].copy(); F["r"] = F.groupby(["dataset", "seed"]).fit.transform(lambda x: rankdata(np.round(x, 12))); print("family SVM", F.groupby("method").r.mean().round(2).to_dict())
for ds in TAB: print(ds, "SVM PERTURB vs RL", pair(S[S.dataset == ds], P, "RL-DOHO")[:3], " KNN", pair(K[K.dataset == ds], P, "RL-DOHO")[:3])
for D, lab in [(S, "SVM"), (K, "KNN")]:
    Q = D[D.dataset.isin(["WDBC", "Dermatology", "SPECTF"])]; print(lab, "3 solid PERTURB vs RL", pair(Q, P, "RL-DOHO")[:4])
# test acc pooled holm SVM
rows = []
tmp = [(c,) + pair(S, "RL-DOHO", c, "test_acc", True)[:4] for c in MS[1:]]
h = holm([x[-1] for x in tmp]); print("test acc SVM min holm", min(h), [(x[0], x[1], x[3], round(hh, 3)) for x, hh in zip(tmp, h)])
tmp = [(c,) + pair(S, "RL-DOHO", c, "fit")[:4] for c in MS[1:]]; h = holm([x[-1] for x in tmp]); print("fit SVM holm", [(x[0], x[1], x[2], x[3], round(hh, 4)) for x, hh in zip(tmp, h)])
print("test PERTURB vs RL SVM", pair(S, P, "RL-DOHO", "test_acc", True)[:4], " KNN", pair(K, P, "RL-DOHO", "test_acc", True)[:4])
print("mean test bal acc SVM vs KNN per ds:", S[S.method.isin(MS)].groupby("dataset").test_acc.mean().round(4).to_dict(), K[K.method.isin(MS)].groupby("dataset").test_acc.mean().round(4).to_dict())
print("friedman test acc per ds SVM:")
for ds in TAB:
    Pm = S[(S.dataset == ds) & S.method.isin(MS)].pivot(index="seed", columns="method", values="test_acc")[MS]; print(" ", ds, round(friedmanchisquare(*[Pm[c] for c in MS]).pvalue, 3))
# all 110+50? overall PERTURB vs RL gene
G = load_all(); print("gene PERTURB vs RL", pair(G[G.group == "gene"], P, "RL-DOHO")[:4], "clinical", pair(G[G.group == "clinical"], P, "RL-DOHO")[:4])
