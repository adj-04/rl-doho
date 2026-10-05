from load2 import *
from scipy.stats import rankdata, binomtest
import os
T="/home/claude/paper/v2final/tex/tables/"; os.makedirs(T,exist_ok=True)
R=load_main(); S=pd.read_csv("summary.csv"); PR=pd.read_csv("pooled.csv"); TIE=1e-12
def pf(p):
    if p<0.001: return r"$<$0.001"
    t=f"{p:.3f}"
    return f"{p:.4f}" if (p<0.05 and float(t)>=0.05) else t
def r0(x): return str(int(np.floor(x+0.5)))
def bold(s,c): return r"\textbf{"+s+"}" if c else s
def wtl(a,b,hb):
    d=np.asarray(a)-np.asarray(b); w=int((d>TIE).sum() if hb else (d<-TIE).sum()); t=int((np.abs(d)<=TIE).sum()); return w,t,len(d)-w-t
# --- rheumatic table
X=R[R.dataset=="Rheumatic"].copy()
for m,hb in [("fit",False),("test_acc",True)]: X["r_"+m]=X.groupby("seed")[m].transform(lambda x: rankdata(-x if hb else x))
g=X.groupby("method").agg(fit=("fit","mean"),fs=("fit","std"),cv=("cv","mean"),acc=("test_acc","mean"),accs=("test_acc","std"),f1=("test_f1","mean"),nf=("n_feats","mean"),rf=("r_fit","mean"),ra=("r_test_acc","mean")).loc[METHODS]
L=[]
for m,r in g.iterrows():
    L.append(" & ".join([r"\textbf{RL-DOHO}" if m=="RL-DOHO" else m, bold(f"{r.fit:.4f}",r.fit==g.fit.min())+f" $\\pm$ {r.fs:.4f}", bold(f"{100*r.cv:.2f}",r.cv==g.cv.max()),
      bold(f"{100*r.acc:.2f}",r.acc==g.acc.max())+f" $\\pm$ {100*r.accs:.2f}", bold(f"{100*r.f1:.2f}",r.f1==g.f1.max()), bold(f"{r.nf:.1f}",r.nf==g.nf.min()),
      bold(f"{r.rf:.2f}",r.rf==g.rf.min()), bold(f"{r.ra:.2f}",r.ra==g.ra.min())])+r" \\")
    if m in ("DOHO","BGWO"): L.append(r"\midrule")
open(T+"rheum.tex","w").write("\n".join(L)); print(g.round(4))
# --- genes table
L=[]
for m in METHODS:
    c=[r"\textbf{RL-DOHO}" if m=="RL-DOHO" else m]
    for ds in DS[1:]:
        Sd=S[S.dataset==ds].set_index("method"); r=Sd.loc[m]
        c+=[bold(f"{r.fit:.3f}",r.fit==Sd.fit.min()),bold(f"{100*r.acc:.1f}",r.acc==Sd.acc.max()),bold(r0(r.nf),r.nf==Sd.nf.min())]
    L.append(" & ".join(c)+r" \\")
    if m in ("DOHO","BGWO"): L.append(r"\midrule")
open(T+"genes.tex","w").write("\n".join(L))
# --- pooled
P=PR[PR.scope=="all"]; L=[]
for c in METHODS[1:]:
    cells=[c]
    for met in ["fit","test_acc","test_f1"]:
        r=P[(P.metric==met)&(P.vs==c)].iloc[0]; cells+=[f"{r.W}/{r['T']}/{r.L}",bold(pf(r.holm),r.holm<0.05)]
    L.append(" & ".join(cells)+r" \\")
    if c in ("DOHO","BGWO"): L.append(r"\midrule")
open(T+"pooled.tex","w").write("\n".join(L))
# --- family table (per-run rank fit) + friedman + first/last
from scipy.stats import friedmanchisquare
Fm=R[R.method.isin(FAMILY)].copy(); Fm["r"]=Fm.groupby(["dataset","seed"]).fit.transform(lambda x: rankdata(x))
L=[]
for ds in DS:
    P2=Fm[Fm.dataset==ds].pivot(index="seed",columns="method",values="fit")[FAMILY]; p=friedmanchisquare(*[P2[c] for c in FAMILY]).pvalue
    rr=Fm[Fm.dataset==ds].groupby("method").r.mean()[FAMILY]
    L.append(" & ".join([ds]+[bold(f"{rr[a]:.2f}",rr[a]==rr.min()) for a in FAMILY]+[pf(p)])+r" \\")
rr=Fm.groupby("method").r.mean()[FAMILY]; L.append(r"\midrule")
L.append(" & ".join(["Mean (60 runs)"]+[bold(f"{rr[a]:.2f}",round(rr[a],2)==round(rr.min(),2)) for a in FAMILY]+["--"])+r" \\")
Fm["rmin"]=Fm.groupby(["dataset","seed"]).fit.rank(method="min"); Fm["rmax"]=Fm.groupby(["dataset","seed"]).fit.rank(method="max")
first=(Fm[Fm.rmin==1].groupby("method").size()/60*100).reindex(FAMILY).fillna(0); last=(Fm[Fm.rmax==4].groupby("method").size()/60*100).reindex(FAMILY).fillna(0)
L.append(" & ".join(["Best or tied (\\%)"]+[f"{first[a]:.0f}" for a in FAMILY]+["--"])+r" \\")
L.append(" & ".join(["Worst or tied (\\%)"]+[bold(f"{last[a]:.0f}",last[a]==last.min()) for a in FAMILY]+["--"])+r" \\")
open(T+"family.tex","w").write("\n".join(L)); print("first",first.to_dict(),"last",last.to_dict())
# --- ablation table: groups rheum / genes ; same budget
A=load_abl("v2_ablation")
A["variant"]=A.variant.replace({"RL-DOHO v2 (UCB, gain reward)":"RL-DOHO (UCB, gain reward)","Dense start (as Notebooks 1-5)":"Dense start","No intervention (DO only)":"No intervention (DO)"})
order=["RL-DOHO (UCB, gain reward)","UCB, old reward","Random arm","PERTURB only","No tie-break","Dense start","No intervention (DO)"]
A["grp"]=np.where(A.dataset=="Rheumatic","R","G")
A["r"]=A.groupby(["dataset","seed"]).fit.transform(lambda x: rankdata(np.round(x,12)))
main="RL-DOHO (UCB, gain reward)"; L=[]; abl={}
for v in order:
    cells=[v]
    for gr in ["R","G"]:
        X=A[A.grp==gr]
        if v not in set(X.variant): cells+=["--","--","--"]; continue
        rk=X.groupby("variant").r.mean(); mf=X.groupby("variant").fit.mean()
        cells+=[f"{mf[v]:.4f}",bold(f"{rk[v]:.2f}",rk[v]==rk.min())]
        if v==main: cells.append("--")
        else:
            Pv=X.pivot_table(index=["dataset","seed"],columns="variant",values="fit"); w,t,l=wtl(Pv[main],Pv[v],False); p=binomtest(w,w+l).pvalue if w+l else 1
            cells.append(f"{w}/{t}/{l} ({pf(p)})"); abl[(v,gr)]=(w,t,l,p)
    L.append(" & ".join(cells)+r" \\")
open(T+"ablation.tex","w").write("\n".join(L)); print(open(T+"ablation.tex").read())
print(A.groupby(["grp","variant"])[["fit","test_acc","n_feats"]].mean().round(4))
# --- long budget table
Lg=load_abl("v2_long"); Lg["grp"]=np.where(Lg.dataset=="Rheumatic","R","G")
lo=["UCB, gain reward","UCB, old reward","Random arm"]; L=[]
for v in lo:
    cells=[v]
    for gr in ["R","G","ALL"]:
        X=Lg if gr=="ALL" else Lg[Lg.grp==gr]
        cells.append(f"{X[X.variant==v].fit.mean():.4f}" if gr!="ALL" else "")
        if v=="UCB, gain reward": cells.append("--")
        else:
            Pv=X.pivot_table(index=["dataset","seed"],columns="variant",values="fit"); w,t,l=wtl(Pv["UCB, gain reward"],Pv[v],False); p=binomtest(w,w+l).pvalue if w+l else 1
            cells.append(f"{w}/{t}/{l} ({pf(p)})")
    cells=[c for c in cells if c!=""]
    L.append(" & ".join(cells)+r" \\")
open(T+"long.tex","w").write("\n".join(L)); print(open(T+"long.tex").read())
print(Lg.groupby(["grp","variant"])[["fit","test_acc","n_feats"]].mean().round(4))
