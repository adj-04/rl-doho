import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/paper/v2final/an")
import load2
R3 = "/home/claude/paper/v3final/res/"
DSN = {"Rheumatic": "Rheumatic", "colon": "Colon", "ALLAML": "ALL/AML", "lung": "Lung", "GLIOMA": "Glioma", "GSE93272": "GSE93272",
       "WDBC": "WDBC", "Hepatitis": "Hepatitis", "Dermatology": "Dermatology", "SPECTF": "SPECTF", "Backache": "Backache"}
GENE = ["Colon", "ALL/AML", "Lung", "Glioma", "GSE93272"]
TAB = ["WDBC", "Hepatitis", "Dermatology", "SPECTF", "Backache"]
DS = ["Rheumatic"] + GENE + TAB
FAMILY = load2.FAMILY; WRAP = load2.WRAP; FILT = load2.FILT; METHODS = load2.METHODS
ADDONS = ["GA+SI", "HO+SI"]
GROUP = {d: ("clinical" if d == "Rheumatic" else "gene" if d in GENE else "tabular") for d in DS}

def load_all():
    """All same-budget runs on the 11 datasets (seeds 0-9), main methods, add-ons and RL-DOHO variants."""
    K = pd.read_csv(R3 + "rl_doho_results_v3_knn/v3_all_runs.csv")
    A = pd.read_csv(R3 + "rl_doho_results_v3_rheum/v3_partA_runs.csv")
    X = pd.concat([A, K], ignore_index=True); X["dataset"] = X.dataset.map(DSN)
    X["group"] = X.dataset.map(GROUP)
    return X[X.seed.between(0, 9)]

def load_long():
    K = pd.read_csv(R3 + "rl_doho_results_v3_knn/v3_long_runs.csv")
    A = pd.read_csv(R3 + "rl_doho_results_v3_rheum/v3_partA_long_runs.csv")
    X = pd.concat([A, K], ignore_index=True); X["dataset"] = X.dataset.map(DSN); X["group"] = X.dataset.map(GROUP)
    return X

def load_splits():
    return pd.read_csv(R3 + "rl_doho_results_v3_rheum/v3_partB_runs.csv")
