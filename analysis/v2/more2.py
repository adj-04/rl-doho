from load2 import *
from scipy.stats import binomtest
T="/home/claude/paper/v2final/tex/tables/"
rh=raw("v2_runs","rl_doho_results_v2_rheum"); ge=raw("v2_runs","rl_doho_results_v2_genes")
def grp_runs(d): return [v for k,v in d.items() if k[1]=="RL-DOHO"]
res={}
for g,lst in [("Clinical",grp_runs(rh)),("Gene",grp_runs(ge))]:
    A=pd.concat([v["arms"] for v in lst]); n=len(lst)
    dec=len(A)/n; init=(A.how=="i").sum()/n
    u=A[A.how=="u"].arm.value_counts(normalize=True)
    # gain decomposition from hist: decreases in best by action type
    rows=[]
    for v in lst:
        H=v["hist"]; b=H.best.values
        for i in range(1,len(H)):
            act=H.action.iloc[i]; a="EXPLORE" if act=="explore" else act.split("(")[0]
            rows.append((a,max(0,b[i-1]-b[i])))
    D=pd.DataFrame(rows,columns=["a","g"]); share=D.groupby("a").g.sum()/D.g.sum()
    succ=A.assign(s=A.gain>1e-12).groupby("arm").s.mean()
    res[g]=dict(dec=dec,init=init,u=u,share=share,succ=succ,mr=A.groupby("arm").reward.mean())
    print(g,round(dec,1),init,u.round(2).to_dict(),share.round(3).to_dict(),succ.round(2).to_dict())
L=[f"Controller decisions per run & {res['Clinical']['dec']:.1f} & {res['Gene']['dec']:.1f}"+r" \\",
   f"\\quad of which initial (one per arm) & {res['Clinical']['init']:.1f} & {res['Gene']['init']:.1f}"+r" \\",
   r"\midrule \multicolumn{3}{l}{\emph{Share of UCB-driven choices (\%)}} \\"]
for a in ["PERTURB","RESTART","DO","HO"]: L.append(f"\\quad {a} & {100*res['Clinical']['u'].get(a,0):.0f} & {100*res['Gene']['u'].get(a,0):.0f}"+r" \\")
L.append(r"\midrule \multicolumn{3}{l}{\emph{Pulls that lowered the best fitness (\%)}} \\")
for a in ["PERTURB","RESTART","DO","HO"]: L.append(f"\\quad {a} & {100*res['Clinical']['succ'].get(a,0):.0f} & {100*res['Gene']['succ'].get(a,0):.0f}"+r" \\")
L.append(r"\midrule \multicolumn{3}{l}{\emph{Share of total fitness reduction (\%)}} \\")
for a,lab in [("EXPLORE","DO exploration steps"),("PERTURB","PERTURB"),("RESTART","RESTART"),("DO","DO arm"),("HO","HO arm")]:
    L.append(f"\\quad {lab} & {100*res['Clinical']['share'].get(a,0):.1f} & {100*res['Gene']['share'].get(a,0):.1f}"+r" \\")
open(T+"arms.tex","w").write("\n".join(L)); print("\n".join(L))
# --- sensitivity v1 vs v2
R2=load_main(); V1=load_v1()
grp=lambda df: np.where(df.dataset=="Rheumatic","Clinical","Gene")
R2["g"]=grp(R2); V1["g"]=grp(V1)
L=[]; 
for m in WRAP:
    c=[m]
    for g in ["Clinical","Gene"]:
        a=V1[(V1.method==m)&(V1.g==g)]; b=R2[(R2.method==m)&(R2.g==g)]
        if g=="Clinical" and m in ["GA","BPSO","BGWO"]:
            c+=[f"{a.fit.mean():.4f}","(same)",f"{100*a.test_acc.mean():.1f}","(same)"]; continue
        c+=[f"{a.fit.mean():.4f}",f"{b.fit.mean():.4f}",f"{100*a.test_acc.mean():.1f}",f"{100*b.test_acc.mean():.1f}"]
    gA=V1[(V1.method==m)&(V1.g=="Gene")].n_feats.mean(); gB=R2[(R2.method==m)&(R2.g=="Gene")].n_feats.mean()
    c+=[str(int(np.floor(gA+0.5))),str(int(np.floor(gB+0.5)))]
    L.append(" & ".join(c)+r" \\")
    if m=="DOHO": L.append(r"\midrule")
open(T+"sens.tex","w").write("\n".join(L)); print("\n".join(L))
# v1 pooled RL-DOHO vs DO/HO for text
for lab,D in [("v1",V1),("v2",R2)]:
    P=D.pivot_table(index=["dataset","seed"],columns="method",values="fit")
    for c in ["DO","HO","DOHO","GA"]:
        d=P["RL-DOHO"]-P[c]; w=(d<-1e-12).sum(); l=(d>1e-12).sum(); t=len(d)-w-l
        print(lab,"vs",c,w,t,l, round(binomtest(w,w+l).pvalue,4))
# --- rheum feature frequency (v2 family + NB4 baselines)
F=['Age','ESR','CRP','RF','Anti-CCP','C3','C4','Gender','HLA-B27','ANA','Anti-Ro','Anti-La','Anti-dsDNA','Anti-Sm']
import pickle
nb4=pickle.load(open(O+"rl_doho_results/baselines_runs_p12_i20_a0.99_tie0.001.pkl","rb"))
freq={a:np.mean([rh[("Rheumatic",a,s)]["mask"] for s in range(10)],axis=0) for a in FAMILY}
for a in ["GA","BPSO","BGWO","LASSO","mRMR","ReliefF"]: freq[a]=np.mean([nb4[(a,s)]["mask"] for s in range(10)],axis=0)
FR=pd.DataFrame(freq,index=F).T; FR.to_csv("featfreq.csv"); print(FR.round(1).to_string())
best=min(v["fit"] for v in rh.values()); print("rheum best v2 run fit", best)
P=load_main(); X=P[P.dataset=="Rheumatic"]; print(X[np.isclose(X.fit,0.18199921647,atol=1e-6)].groupby("method").size())
