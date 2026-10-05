from common3 import *
import load2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = "/home/claude/paper/v3final/tex/figs/"
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8,
 "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.linewidth": 0.6,
 "pdf.fonttype": 42, "font.family": "serif", "mathtext.fontset": "dejavuserif", "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
X = load_all(); L = load_long(); S = pd.read_csv("summary3.csv")
# ---------- CD diagrams over 11 datasets
def cd_plot(ax, metric, hb, title):
    P = S.pivot(index="dataset", columns="method", values=metric).loc[DS, METHODS]
    Rk = P.apply(lambda r: pd.Series(rankdata(-r.values if hb else r.values), index=r.index), axis=1).mean().sort_values()
    k = len(METHODS); N = len(DS); cd = 3.219 * np.sqrt(k * (k + 1) / (6 * N)); lo, hi = 1, k
    ax.set_xlim(lo - 0.3, hi + 0.3); ax.set_ylim(0.27, 1.12); ax.axis("off"); y0 = 0.80
    ax.plot([lo, hi], [y0, y0], color="k", lw=0.8)
    for t in range(lo, hi + 1): ax.plot([t, t], [y0, y0 + 0.03], color="k", lw=0.7); ax.text(t, y0 + 0.05, str(t), ha="center", va="bottom", fontsize=6.5)
    ax.plot([lo, lo + cd], [0.97, 0.97], color="#b02a2a", lw=1.6); ax.text(lo + cd / 2, 0.985, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=6.2, color="#b02a2a")
    ax.text((lo + hi) / 2, 1.10, title, ha="center", va="bottom", fontsize=7.5)
    names = list(Rk.index); half = int(np.ceil(len(names) / 2))
    for i, n in enumerate(names):
        x = Rk[n]; left = i < half; yy = y0 - 0.09 - 0.075 * (i if left else (len(names) - 1 - i)); xt = lo - 0.2 if left else hi + 0.2
        c = "#c07c00" if n == "RL-DOHO" else "k"
        ax.plot([x, x, xt], [y0, yy, yy], color=c, lw=1.1 if n == "RL-DOHO" else 0.7)
        ax.text(xt + (-0.05 if left else 0.05), yy, f"{n} ({x:.2f})", ha="right" if left else "left", va="center", fontsize=6.2, color=c, fontweight="bold" if n == "RL-DOHO" else "normal")
    v = Rk.values; groups = []
    for i in range(len(v)):
        j = i
        while j + 1 < len(v) and v[j + 1] - v[i] < cd: j += 1
        if j > i and not any(a <= i and b >= j for a, b in groups): groups.append((i, j))
    for gi, (i, j) in enumerate(groups): ax.plot([v[i] - 0.04, v[j] + 0.04], [y0 - 0.03 - 0.03 * gi] * 2, color="#444", lw=1.3, solid_capstyle="round")
    return Rk.round(2).to_dict(), round(cd, 2), groups
fig, axs = plt.subplots(2, 1, figsize=(7.0, 2.7))
for ax, (m, hb, t) in zip(axs, [("fit", False, "(a) Fitness (lower is better)"), ("acc", True, "(b) Test accuracy (higher is better)")]): print(cd_plot(ax, m, hb, t))
fig.subplots_adjust(hspace=0.25, left=0.12, right=0.88); fig.savefig(OUT + "cd.pdf"); plt.close()
# ---------- ablation ranks: three groups same budget + 3x
VMAP = {"RL-DOHO": "RL-DOHO (UCB1)", "RL-DOHO [Q-learning]": "Q-learning", "RL-DOHO [random arm]": "Random arm",
        "RL-DOHO [PERTURB only]": "PERTURB only", "RL-DOHO [no intervention]": "No intervention (DO)"}
VC = {"RL-DOHO (UCB1)": "#eda100", "Q-learning": "#4a3aa7", "Random arm": "#c9c8c0", "PERTURB only": "#1baf7a", "No intervention (DO)": "#2a78d6"}
A = X[X.method.isin(VMAP)].copy(); A["v"] = A.method.map(VMAP); A["r"] = A.groupby(["dataset", "seed"]).fit.transform(lambda x: rankdata(np.round(x, 12)))
LL = L.copy(); LL["v"] = LL.method.map({"UCB1 (3x)": "RL-DOHO (UCB1)", "Q-learning (3x)": "Q-learning", "Random arm (3x)": "Random arm"})
LL["r"] = LL.groupby(["dataset", "seed"]).fit.transform(lambda x: rankdata(np.round(x, 12)))
order = list(VMAP.values())
panels = [("Clinical (10 runs)", A[A.group == "clinical"]), ("Gene expression (50)", A[A.group == "gene"]), ("Medical tabular (50)", A[A.group == "tabular"]), ("All, 3$\\times$ budget (110)", LL)]
fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.85), sharey=True)
for ax, (t, D) in zip(axs, panels):
    rk = D.groupby("v").r.mean().reindex(order)
    for i, (v, x) in enumerate(rk.items()):
        y = len(order) - 1 - i
        if np.isnan(x): ax.text(0.1, y, "not run", va="center", fontsize=6, color="#888"); continue
        ax.barh(y, x, color=VC[v], height=0.65, edgecolor="white", lw=0.8); ax.text(x + 0.05, y, f"{x:.2f}", va="center", fontsize=6)
    ax.set_title(t); ax.set_xlabel("Mean rank"); ax.grid(axis="y", alpha=0); ax.set_xlim(0, 5.4)
axs[0].set_yticks(range(len(order))[::-1]); axs[0].set_yticklabels(order)
fig.tight_layout(w_pad=0.5); fig.savefig(OUT + "ablation.pdf"); plt.close()
# ---------- add-on and kappa figure
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.15), gridspec_kw=dict(width_ratios=[1.7, 1]))
ax = axs[0]
P = X.pivot_table(index="dataset", columns="method", values="fit")
lab = {"Colon": "Colon", "ALL/AML": "ALL/AML"}
xs = np.arange(len(DS))
for k, (a, b, c, mk) in enumerate([("HO+SI", "HO", "#eb6834", "s"), ("GA+SI", "GA", "#4a3aa7", "v")]):
    d = [100 * (P.loc[ds, a] - P.loc[ds, b]) / P.loc[ds, b] for ds in DS]
    ax.scatter(xs + (k - 0.5) * 0.22, d, color=c, marker=mk, s=20, label=f"{a} vs. {b}", zorder=3)
    for x_, y_ in zip(xs + (k - 0.5) * 0.22, d): ax.plot([x_, x_], [0, y_], color=c, lw=0.9, alpha=0.6)
ax.axhline(0, color="k", lw=0.7); ax.set_xticks(xs); ax.set_xticklabels(DS, rotation=45, ha="right", fontsize=6.3)
ax.set_ylabel("Change in mean fitness (%)"); ax.set_ylim(-19, 24); ax.set_title("(a) Effect of the intervention layer (below 0 = better)"); ax.legend(frameon=False, loc="upper left", ncol=2)
for xv in (0.5, 5.5): ax.axvline(xv, color="#999", lw=0.5, ls=":")
ax = axs[1]
KAP = {"RL-DOHO [κ=1]": 1, "RL-DOHO": 2, "RL-DOHO [κ=3]": 3, "RL-DOHO [κ=4]": 4}
K = X[X.method.isin(KAP)].copy(); K["k"] = K.method.map(KAP); K["r"] = K.groupby(["dataset", "seed"]).fit.transform(lambda x: rankdata(x))
for g, c, mk in [("clinical", "#2a78d6", "o"), ("gene", "#1baf7a", "^"), ("tabular", "#e87ba4", "D")]:
    rk = K[K.group == g].groupby("k").r.mean(); ax.plot(rk.index, rk.values, marker=mk, color=c, lw=1.2, ms=3.5, label={"clinical": "Clinical", "gene": "Gene", "tabular": "Tabular"}[g])
rk = K.groupby("k").r.mean(); ax.plot(rk.index, rk.values, color="k", lw=2, marker="o", ms=4, label="All 110 runs")
ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("Stagnation trigger $\\kappa$"); ax.set_ylabel("Mean rank among the four $\\kappa$"); ax.set_title("(b) Sensitivity to $\\kappa$")
ax.legend(frameon=False, fontsize=6, ncol=2)
fig.tight_layout(w_pad=1.2); fig.savefig(OUT + "addon_kappa.pdf"); plt.close()
print("figs ok")
