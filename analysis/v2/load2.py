import pickle, numpy as np, pandas as pd
B="/home/claude/paper/v2final/b/"; O="/home/claude/paper/bundle/"
DSN={"Rheumatic":"Rheumatic","colon":"Colon","ALLAML":"ALL/AML","lung":"Lung","GLIOMA":"Glioma","GSE93272":"GSE93272"}
DS=["Rheumatic","Colon","ALL/AML","Lung","Glioma","GSE93272"]
FAMILY=["RL-DOHO","DO","HO","DOHO"]; WRAP=FAMILY+["GA","BPSO","BGWO"]; FILT=["LASSO","mRMR","ReliefF","All features"]
METHODS=WRAP+FILT
cols=["fit","cv","n_feats","test_acc","test_prec","test_rec","test_f1"]
def _rows(d, key, rename_cv=False):
    out=[]
    for k,v in d.items():
        ds,m,s=key(k)
        r={"dataset":DSN[ds],"method":m,"seed":s}
        for c in cols: r[c]=v.get(c, v.get("cv_acc") if c=="cv" else np.nan)
        out.append(r)
    return out
def load_main():
    rows=[]
    rh=pickle.load(open(B+"rl_doho_results_v2_rheum/v2_runs.pkl","rb")); rows+=_rows(rh,lambda k:k)
    nb4=pickle.load(open(O+"rl_doho_results/baselines_runs_p12_i20_a0.99_tie0.001.pkl","rb"))
    rows+=_rows({k:v for k,v in nb4.items() if k[0] in ["GA","BPSO","BGWO"]+FILT}, lambda k:("Rheumatic",k[0],k[1]))
    g=pickle.load(open(B+"rl_doho_results_v2_genes/v2_runs.pkl","rb")); rows+=_rows(g,lambda k:k)
    nb5=pickle.load(open(O+"rl_doho_results_baselines_hd/baselines_hd_runs_p20_i50.pkl","rb"))
    rows+=_rows({k:v for k,v in nb5.items() if k[1] in FILT}, lambda k:k)
    return pd.DataFrame(rows)
def load_v1():
    rows=[]
    nb4=pickle.load(open(O+"rl_doho_results/baselines_runs_p12_i20_a0.99_tie0.001.pkl","rb"))
    rows+=_rows({k:v for k,v in nb4.items() if k[0] in WRAP}, lambda k:("Rheumatic",k[0],k[1]))
    nb5=pickle.load(open(O+"rl_doho_results_baselines_hd/baselines_hd_runs_p20_i50.pkl","rb"))
    rows+=_rows({k:v for k,v in nb5.items() if k[1] in WRAP}, lambda k:k)
    return pd.DataFrame(rows)
def load_abl(kind):   # kind: v2_ablation or v2_long
    rows=[]
    for f in ["rl_doho_results_v2_rheum","rl_doho_results_v2_genes"]:
        d=pickle.load(open(B+f+f"/{kind}.pkl","rb")); rows+=[dict(r, variant=r.pop("method")) for r in _rows(d,lambda k:k)]
    A=pd.DataFrame(rows)
    if kind=="v2_ablation":
        M=load_main(); M=M[M.method=="RL-DOHO"].rename(columns={"method":"variant"}); M["variant"]="RL-DOHO v2 (UCB, gain reward)"
        A=pd.concat([A,M],ignore_index=True)
    return A
def raw(kind, folder):
    return pickle.load(open(B+folder+f"/{kind}.pkl","rb"))
