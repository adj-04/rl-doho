from common3 import *
X = load_all(); L = load_long(); Y = X[~X.dataset.isin(["Backache", "Hepatitis"])]; LY = L[~L.dataset.isin(["Backache", "Hepatitis"])]
print("mean test bal acc over 11 methods:", X[X.method.isin(METHODS) & X.dataset.isin(["Backache", "Hepatitis"])].groupby("dataset").test_acc.mean().round(3).to_dict())
rows = [("RL-DOHO vs.\\ DO", "RL-DOHO", "DO", X, Y), ("RL-DOHO vs.\\ HO", "RL-DOHO", "HO", X, Y), ("RL-DOHO vs.\\ DOHO", "RL-DOHO", "DOHO", X, Y),
        ("RL-DOHO vs.\\ GA", "RL-DOHO", "GA", X, Y), ("RL-DOHO vs.\\ BPSO", "RL-DOHO", "BPSO", X, Y), ("RL-DOHO vs.\\ mRMR", "RL-DOHO", "mRMR", X, Y),
        ("HO+SI vs.\\ HO", "HO+SI", "HO", X, Y), ("GA+SI vs.\\ GA", "GA+SI", "GA", X, Y),
        ("PERTURB only vs.\\ RL-DOHO", "RL-DOHO [PERTURB only]", "RL-DOHO", X, Y),
        ("Q-learning vs.\\ random arm, 3$\\times$", "Q-learning (3x)", "Random arm (3x)", L, LY)]
out = []
for lab, a, b, A, B in rows:
    c = [lab]
    for D in (A, B):
        w, t, l, p, n, _ = pair(D, a, b); c += [f"{w}/{t}/{l}", pf(p)]
    out.append(" & ".join(c) + r" \\"); print(out[-1])
KAP = {"RL-DOHO [κ=1]": 1, "RL-DOHO": 2, "RL-DOHO [κ=3]": 3, "RL-DOHO [κ=4]": 4}
ps = []
for D in (X, Y):
    P = D[D.method.isin(KAP)].assign(k=lambda d: d.method.map(KAP)).pivot_table(index=["dataset", "seed"], columns="k", values="fit")
    ps.append(friedmanchisquare(*[P[c] for c in P]).pvalue)
out.append(r"$\kappa\in\{1,2,3,4\}$ (Friedman) & -- & " + pf(ps[0]) + " & -- & " + pf(ps[1]) + r" \\")
open(T + "weak.tex", "w").write("\n".join(out)); print(out[-1])
