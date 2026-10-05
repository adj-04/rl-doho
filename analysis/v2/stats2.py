from load2 import *
from scipy.stats import friedmanchisquare, wilcoxon, binomtest, rankdata
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
TIE=1e-12
def holm(ps):
    ps=np.asarray(ps,float); o=np.argsort(ps); m=len(ps); adj=np.empty(m); run=0
    for i,j in enumerate(o): run=max(run,(m-i)*ps[j]); adj[j]=min(1,run)
    return adj
def wtl(a,b,hb):
    d=np.asarray(a)-np.asarray(b); w=int((d>TIE).sum() if hb else (d<-TIE).sum()); t=int((np.abs(d)<=TIE).sum()); return w,t,len(d)-w-t
R=load_main()
S=R.groupby(["dataset","method"]).agg(fit=("fit","mean"),fit_sd=("fit","std"),cv=("cv","mean"),acc=("test_acc","mean"),acc_sd=("test_acc","std"),f1=("test_f1","mean"),nf=("n_feats","mean")).reset_index()
S.to_csv("summary.csv",index=False)
# per dataset tests
rows=[]; fr=[]
for ds in DS:
    for m,hb in [("fit",False),("test_acc",True),("test_f1",True)]:
        P=R[R.dataset==ds].pivot(index="seed",columns="method",values=m)[METHODS]
        fr.append((ds,m,*friedmanchisquare(*[P[c] for c in METHODS])))
        tmp=[]
        for c in METHODS[1:]:
            w,t,l=wtl(P["RL-DOHO"],P[c],hb)
            try: p=wilcoxon(P["RL-DOHO"],P[c],zero_method="zsplit").pvalue
            except ValueError: p=1.0
            tmp.append([ds,m,c,w,t,l,(P["RL-DOHO"]-P[c]).mean(),p])
        for x,h in zip(tmp,holm([x[-1] for x in tmp])): rows.append(x+[h])
W=pd.DataFrame(rows,columns=["dataset","metric","vs","W","T","L","md","p","holm"]); W.to_csv("wilcoxon.csv",index=False)
F=pd.DataFrame(fr,columns=["dataset","metric","chi2","p"]); F.to_csv("friedman.csv",index=False)
print(F.round(5).to_string())
print(W[W.holm<0.05].round(4).to_string())
# pooled
pr=[]
for scope,dss in [("all",DS),("genes",DS[1:])]:
  for m,hb in [("fit",False),("test_acc",True),("test_f1",True)]:
    tmp=[]
    for c in METHODS[1:]:
        A=R[(R.method=="RL-DOHO")&R.dataset.isin(dss)].set_index(["dataset","seed"])[m]; Bm=R[(R.method==c)&R.dataset.isin(dss)].set_index(["dataset","seed"])[m]
        a,b=A.align(Bm,join="inner"); w,t,l=wtl(a,b,hb); p=binomtest(w,w+l).pvalue if w+l else 1
        tmp.append([scope,m,c,w,t,l,p])
    for x,h in zip(tmp,holm([x[-1] for x in tmp])): pr.append(x+[h])
PR=pd.DataFrame(pr,columns=["scope","metric","vs","W","T","L","p","holm"]); PR.to_csv("pooled.csv",index=False)
print(PR[PR.scope=="all"].round(4).to_string())
# dataset-level ranks
for m,hb in [("fit",False),("acc",True)]:
    P=S.pivot(index="dataset",columns="method",values=m).loc[DS,METHODS]
    Rk=P.apply(lambda r: pd.Series(rankdata(-r.values if hb else r.values),index=r.index),axis=1)
    st,p=friedmanchisquare(*[P[c] for c in METHODS]); print(f"\ndataset-level {m}: chi2={st:.2f} p={p:.4g}", Rk.mean().sort_values().round(2).to_dict())
# family per-run ranks
Fm=R[R.method.isin(FAMILY)].copy()
for m,hb in [("fit",False),("test_acc",True)]:
    Fm["r"]=Fm.groupby(["dataset","seed"])[m].transform(lambda x: rankdata(-x if hb else x))
    T=Fm.pivot_table(index="dataset",columns="method",values="r").loc[DS,FAMILY]
    print(f"\nfamily per-run rank {m}:\n",T.round(2).to_string(),"\nall:",Fm.groupby("method").r.mean().reindex(FAMILY).round(2).to_dict())
    print(" first%:",(Fm[Fm.r==1].groupby("method").size()/60*100).reindex(FAMILY).round(0).to_dict()," last%:",(Fm[Fm.r==4].groupby("method").size()/60*100).reindex(FAMILY).fillna(0).round(0).to_dict())
    if m=="fit":
        fpp=[friedmanchisquare(*[Fm[Fm.dataset==ds].pivot(index="seed",columns="method",values="fit")[c] for c in FAMILY]).pvalue for ds in DS]; print(" family friedman:",dict(zip(DS,np.round(fpp,4))))
P=R[R.method.isin(FAMILY)].pivot_table(index=["dataset","seed"],columns="method",values="fit"); gap=P.sub(P.min(axis=1),axis=0)
print("gap to family best mean:",gap.mean().reindex(FAMILY).round(4).to_dict(),"max:",gap.max().reindex(FAMILY).round(4).to_dict())
