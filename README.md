# RL-DOHO: stagnation-triggered interventions for wrapper feature selection

Code, notebooks, per-run results and paper for:

> **Stagnation-Triggered Interventions for Wrapper Feature Selection: RL-DOHO and a Controlled Test of Learned Operator Selection**
> Preetha Evangeline D, Prasenjit Choudhury, Shreyansh Dabgar, Aditya Jha. School of Computer Science and Engineering, VIT Chennai.
> Final-year project (BCSE497J). Preprint: [`paper/main.pdf`](paper/main.pdf). Supplement: [`paper/supplementary.pdf`](paper/supplementary.pdf).

## What this is

Population-based wrapper feature selection tends to stop improving well before its evaluation budget runs out. RL-DOHO addresses this with a **stagnation-triggered intervention layer**:

1. Run the published **Dandelion Optimizer (DO)** while the best subset keeps improving.
2. After **κ = 2** iterations without improvement, a **UCB1 bandit** chooses one intervention:
   - **PERTURB**: flip 1–3 features of the best subset (tabu-guarded).
   - **RESTART**: redraw the worse half of the population (tabu-guarded).
   - One published **DO** iteration.
   - One published **Hippopotamus Optimization (HO)** iteration.
3. The bandit's reward is the fitness gain per evaluation spent. A sparse initial population is used when D > 100.

Fitness is `0.99 × (1 − CV accuracy) + 0.01 × |S|/D`. Every method gets the same budget of **fitness evaluations**.

## Main findings

The study covers 11 datasets and 110 paired runs per comparison. Ten hypotheses were fixed before the runs that test them.

| Finding | Result |
|---|---|
| RL-DOHO vs published DO | better in 103 runs, worse in 4 |
| RL-DOHO vs published HO | level (58 vs 49, not significant) |
| GA vs RL-DOHO | GA better in 85 runs, worse in 22 |
| Same intervention layer added to HO | better in 71 runs, worse in 29 (p < 0.001) |
| Same layer added to a GA | no change (45 vs 49) |
| UCB1 / Q-learning / random choice of intervention | level at 1× and 3× budget |
| Warm-started UCB1 vs random (H10a) | level (46 vs 53) |
| UCB1 with best-subset reward vs random (H11a/b) | level (50 vs 48; 55 vs 42) |
| Trigger κ | κ ≤ 2 better than waiting (Friedman p < 0.001) |
| Test accuracy | no difference significant after correction |
| SVM instead of 5-NN (tabular) | method ranking barely changes (Spearman 0.99) |

**What this means:** the gain comes from intervening when the search stalls, not from how the intervention is chosen.

**Why the learned choice does not help:** with the original reward the bandit learns that reward exactly, but it favours DO steps that rarely improve the best subset. With a reward aimed at the best subset (Notebook 12) it learns to favour the right moves, yet still does not beat random choice: even the best move improves the best subset in at most 28% of uses. See Section VI-C and Supplementary S3 and S5 of the paper.

## Datasets

| Group | Datasets | Features | Classifier in the fitness |
|---|---|---|---|
| Clinical | Rheumatic and autoimmune disease, 12,085 patients, 7 classes (Mahdi et al., 2025) | 14 | XGBoost + HistGradientBoosting, SMOTE inside 3-fold CV |
| Gene expression | colon, ALL/AML, lung, GLIOMA (scikit-feature), GSE93272 (GEO) | 2,000–7,129 | 5-NN |
| Medical tabular | WDBC, Hepatitis, Dermatology, SPECTF, Backache (PMLB) | 19–44 | 5-NN, and an SVM in Notebook 10 |

Raw data files are not stored here (see `.gitignore`). The notebooks download them from the original sources or read them from your Google Drive.

## Notebooks

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_rheumatic_main.ipynb` | First version on the rheumatic data |
| 02 | `02_gene_expression.ipynb` | First version on the gene-expression data |
| 03 | `03_gse93272_feature_sweep.ipynb` | Probe-filter sweep for GSE93272 |
| 04 | `04_rheumatic_baselines.ipynb` | GA, BPSO, BGWO and filter baselines (rheumatic) |
| 05 | `05_gene_baselines.ipynb` | Same baselines on gene data |
| 06 | `06_v2_rheumatic.ipynb` | v2: published DO/HO rules, gain reward, equal budgets (rheumatic) |
| 07 | `07_v2_gene_expression.ipynb` | v2 on gene data, with the sparse start |
| 08 | `08_v3_gene_and_tabular_extensions.ipynb` | Tabular datasets, GA+SI / HO+SI, Q-learning, κ sweep, 3× budget, cost |
| 09 | `09_v3_rheumatic_extensions.ipynb` | Extensions on the rheumatic data and two extra train/test splits |
| 10 | `10_v4_svm_tabular.ipynb` | Tabular benchmark repeated with an SVM (H9) |
| 11 | `11_v5_controller_warm_start.ipynb` | Warm-started controller vs random choice (H10a), with reward diagnostics |
| 12 | `12_v6_best_gain_reward.ipynb` | Reward credited only for improving the best subset, with c = 1 and c = 0.2 (H11) |

Notebook 11 also contains a 7-move pool (with drop and swap moves) and a LinUCB contextual bandit. They are fixed in advance as H10b–d but have not been run; set `ONLY = None` to run them.

## Running the notebooks

- They are written for **Google Colab** with Google Drive mounted. Results go to `MyDrive/rl_doho_results_*` in Drive; copies are in `results/` here.
- Every run is saved as soon as it finishes, so rerunning a cell resumes where it stopped.
- Set `FAST = True` in the config cell for a quick check that a notebook runs.
- Later notebooks reuse the fitness caches of earlier ones. The fitness is deterministic, so cached and fresh evaluations are identical.
- Rough runtimes on free Colab:
  - Notebook 10: 1–1.5 h.
  - Notebook 11 (H10a only): 1–1.5 h.
  - Notebook 9: several hours, because each clinical evaluation takes about 3.6 s.

## Repository layout

```
notebooks/   the 12 notebooks, with outputs
paper/       main.pdf, supplementary.pdf, LaTeX sources (main.tex, supp.tex, refs.tex), figs/, tables/
results/     per-run results (pickles and CSVs), one folder per version (v1_paper_bundle ... v5_rl)
```

## Citation

```
P. Evangeline D, P. Choudhury, S. Dabgar, A. Jha, "Stagnation-Triggered Interventions for Wrapper
Feature Selection: RL-DOHO and a Controlled Test of Learned Operator Selection," preprint, 2026.
https://github.com/adj-04/rl-doho
```
