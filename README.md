# RL-DOHO

A Dandelion–Hippopotamus hybrid with stagnation-triggered interventions for wrapper feature selection.
Final Year Project-I (BCSE497J), School of Computer Science and Engineering, VIT Chennai.

**Authors:** Preetha Evangeline D, Prasenjit Choudhury, Shreyansh Dabgar, Aditya Jha

Preprint: [`paper/main.pdf`](paper/main.pdf)

## Repository layout
| Folder | Contents |
|---|---|
| `notebooks/` | All experiments, in run order (Google Colab) |
| `code/` | `rl_doho_core_v3.py`: DO, HO, DOHO, RL-DOHO, GA, BPSO, BGWO, GA+SI, HO+SI, controllers, filters |
| `analysis/` | Scripts that turn per-run results into the paper's tables and figures |
| `results/` | Per-run results (`.pkl` / `.csv`) for every notebook |
| `paper/` | LaTeX source, figures and PDF |

## Notebooks
| # | Notebook | What it does |
|---|---|---|
| 1 | `01_rheumatic_main` | First DO / HO / DOHO / RL-DOHO comparison, rheumatic data (simplified operators) |
| 2 | `02_gene_expression` | Same on four cancer microarray datasets |
| 3 | `03_gse93272_feature_sweep` | RA whole-blood expression (GSE93272), feature-count sweep |
| 4 | `04_rheumatic_baselines` | GA, BPSO, BGWO, LASSO, mRMR, ReliefF on rheumatic data |
| 5 | `05_gene_baselines` | Same baselines on the five gene-expression datasets |
| 6 | `06_v2_rheumatic` | Final version: published DO/HO, gain reward; ablation, 3x budget |
| 7 | `07_v2_gene_expression` | Final version on gene data, sparse start; ablation, 3x budget |
| 8 | `08_v3_gene_and_tabular_extensions` | Five medical tabular datasets, GA+SI / HO+SI, Q-learning, kappa sweep, cost |
| 9 | `09_v3_rheumatic_extensions` | Same extensions on rheumatic data, plus repeated train/test splits |

## Data
- Rheumatic and autoimmune disease dataset (CC BY 4.0): Mahdi, Jahani & Abd, *Data in Brief* 60 (2025) 111623, doi:10.1016/j.dib.2025.111623. Put the `.xlsx` in your Google Drive root before running Notebooks 1, 4, 6 and 9.
- Microarray datasets: scikit-feature repository (downloaded automatically).
- GSE93272: NCBI GEO (downloaded automatically).
- WDBC, Hepatitis, Dermatology, SPECTF, Backache: PMLB (downloaded automatically).

## Running
Open a notebook in Google Colab and run all cells. Results are saved to Google Drive after every run, so an interrupted notebook resumes where it stopped.
