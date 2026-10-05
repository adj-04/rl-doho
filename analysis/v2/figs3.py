from load2 import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import rankdata
OUT="/home/claude/paper/v2final/tex/figs/"
COL={"DO":"#2a78d6","HO":"#eb6834","DOHO":"#1baf7a","RL-DOHO":"#eda100"}; MK={"DO":"o","HO":"s","DOHO":"^","RL-DOHO":"D"}
plt.rcParams.update({"font.size":7.5,"axes.titlesize":8,"axes.labelsize":7.5,"xtick.labelsize":6.8,"ytick.labelsize":6.8,"legend.fontsize":6.8,
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,"grid.linewidth":0.5,"axes.linewidth":0.6,
 "pdf.fonttype":42,"font.family":"serif","mathtext.fontset":"dejavuserif","savefig.bbox":"tight","savefig.pad_inches":0.02})
rh=raw("v2_runs","rl_doho_results_v2_rheum"); ge=raw("v2_runs","rl_doho_results_v2_genes")
R=load_main(); S=pd.read_csv("summary.csv")
def mean_curve(get,budget,n=120):
    grid=np.linspace(0,budget,n)
    return grid,{a:np.mean([np.interp(grid,get(a,s)["hist"]["evals"],get(a,s)["hist"]["best"]) for s in range(10)],0) for a in FAMILY}
# 1 rheum convergence
g,C=mean_curve(lambda a,s: rh[("Rheumatic",a,s)],252)
fig,ax=plt.subplots(figsize=(3.4,2.15))
for a in ["DO","HO","DOHO","RL-DOHO"]:
    ax.plot(g,C[a],color=COL[a],lw=2.2 if a=="RL-DOHO" else 1.3,marker=MK[a],ms=3,markevery=20,label=a,zorder=3 if a=="RL-DOHO" else 2)
ax.set_xlabel("Fitness evaluations"); ax.set_ylabel("Mean best fitness (10 seeds)"); ax.set_xlim(0,252); ax.legend(frameon=False,ncol=2)
fig.savefig(OUT+"conv_rheum.pdf"); plt.close()
# 2 gene convergence
panels=[("colon","Colon"),("ALLAML","ALL/AML"),("lung","Lung"),("GLIOMA","Glioma"),("GSE93272","GSE93272")]
fig,axs=plt.subplots(1,5,figsize=(7.16,1.75))
for k,(key,lab) in enumerate(panels):
    g,C=mean_curve(lambda a,s: ge[(key,a,s)],1020); ax=axs[k]
    for a in ["DO","HO","DOHO","RL-DOHO"]:
        ax.plot(g,C[a],color=COL[a],lw=2.0 if a=="RL-DOHO" else 1.1,marker=MK[a],ms=2.6,markevery=24,label=a,zorder=3 if a=="RL-DOHO" else 2)
    ax.set_title(lab); ax.set_xlabel("Evaluations"); ax.set_xticks([0,500,1000]); ax.tick_params(axis="y",labelsize=6)
axs[0].set_ylabel("Mean best fitness"); h_,l_=axs[0].get_legend_handles_labels()
fig.legend(h_,l_,ncol=4,frameon=False,loc="upper center",bbox_to_anchor=(0.5,1.1)); fig.tight_layout(w_pad=0.6); fig.savefig(OUT+"conv_genes.pdf"); plt.close()
# 3 CD diagrams
def cd_plot(ax,metric,hb,title):
    P=S.pivot(index="dataset",columns="method",values=metric).loc[DS,METHODS]
    Rk=P.apply(lambda r: pd.Series(rankdata(-r.values if hb else r.values),index=r.index),axis=1).mean().sort_values()
    k=len(METHODS); N=len(DS); cd=3.219*np.sqrt(k*(k+1)/(6*N)); lo,hi=1,k
    ax.set_xlim(lo-0.3,hi+0.3); ax.set_ylim(0.27,1.12); ax.axis("off"); y0=0.80
    ax.plot([lo,hi],[y0,y0],color="k",lw=0.8)
    for t in range(lo,hi+1): ax.plot([t,t],[y0,y0+0.03],color="k",lw=0.7); ax.text(t,y0+0.05,str(t),ha="center",va="bottom",fontsize=6.5)
    ax.plot([lo,lo+cd],[0.97,0.97],color="#b02a2a",lw=1.6); ax.text(lo+cd/2,0.985,f"CD = {cd:.2f}",ha="center",va="bottom",fontsize=6.2,color="#b02a2a")
    ax.text((lo+hi)/2,1.10,title,ha="center",va="bottom",fontsize=7.5)
    names=list(Rk.index); half=int(np.ceil(len(names)/2))
    for i,n in enumerate(names):
        x=Rk[n]; left=i<half; yy=y0-0.09-0.075*(i if left else (len(names)-1-i)); xt=lo-0.2 if left else hi+0.2
        c="#c07c00" if n=="RL-DOHO" else "k"
        ax.plot([x,x,xt],[y0,yy,yy],color=c,lw=1.1 if n=="RL-DOHO" else 0.7)
        ax.text(xt+(-0.05 if left else 0.05),yy,f"{n} ({x:.2f})",ha="right" if left else "left",va="center",fontsize=6.2,color=c,fontweight="bold" if n=="RL-DOHO" else "normal")
    v=Rk.values; groups=[]
    for i in range(len(v)):
        j=i
        while j+1<len(v) and v[j+1]-v[i]<cd: j+=1
        if j>i and not any(a<=i and b>=j for a,b in groups): groups.append((i,j))
    for gi,(i,j) in enumerate(groups): ax.plot([v[i]-0.04,v[j]+0.04],[y0-0.03-0.03*gi]*2,color="#444",lw=1.3,solid_capstyle="round")
    return Rk,cd,groups
fig,axs=plt.subplots(2,1,figsize=(7.0,2.7))
for ax,(m,hb,t) in zip(axs,[("fit",False,"(a) Fitness (lower is better)"),("acc",True,"(b) Test accuracy (higher is better)")]): print(cd_plot(ax,m,hb,t))
fig.subplots_adjust(hspace=0.25,left=0.12,right=0.88); fig.savefig(OUT+"cd.pdf"); plt.close()
# 4 accuracy vs size (genes)
other={"GA":("#4a3aa7","v"),"BPSO":("#e87ba4","P"),"BGWO":("#008300","X"),"LASSO":("#555555","<"),"mRMR":("#555555",">"),"ReliefF":("#555555","h"),"All features":("#999999","*")}
fig,axs=plt.subplots(1,5,figsize=(7.16,1.85))
for k,ds in enumerate(DS[1:]):
    ax=axs[k]; Sd=S[S.dataset==ds].set_index("method")
    for m in METHODS:
        c,mk=(COL[m],MK[m]) if m in COL else other[m]; filt=m in ("LASSO","mRMR","ReliefF")
        ax.scatter(Sd.loc[m,"nf"],Sd.loc[m,"acc"]*100,s=26 if m=="RL-DOHO" else 16,color="white" if filt else c,edgecolor=c,linewidth=0.9,marker=mk,label=m,zorder=4 if m=="RL-DOHO" else 3)
    ax.set_xscale("log"); ax.set_title(ds); ax.set_xlabel("Selected features (log)")
axs[0].set_ylabel("Mean test accuracy (%)"); h_,l_=axs[0].get_legend_handles_labels()
fig.legend(h_,l_,ncol=11,frameon=False,loc="upper center",bbox_to_anchor=(0.5,1.12),handletextpad=0.1,columnspacing=0.7)
fig.tight_layout(w_pad=0.5); fig.savefig(OUT+"acc_vs_size.pdf"); plt.close()
# 5 rheum boxplots
mcol=lambda m: COL.get(m,"#7a7a7a" if m in ("LASSO","mRMR","ReliefF","All features") else "#4a3aa7")
X=R[R.dataset=="Rheumatic"]
fig,axs=plt.subplots(1,3,figsize=(7.16,2.1))
for ax,(m,lab,sc) in zip(axs,[("fit","Final fitness (lower is better)",1),("test_acc","Test accuracy (%)",100),("n_feats","Selected features",1)]):
    data=[X[X.method==x][m].values*sc for x in METHODS]
    bp=ax.boxplot(data,widths=0.55,patch_artist=True,showfliers=False,medianprops=dict(color="k",lw=0.9),whiskerprops=dict(lw=0.7),capprops=dict(lw=0.7),boxprops=dict(lw=0.6))
    for p,x in zip(bp["boxes"],METHODS): p.set_facecolor(mcol(x)); p.set_alpha(0.55 if x!="RL-DOHO" else 0.9)
    rng=np.random.default_rng(0)
    for i,v in enumerate(data): ax.scatter(i+1+rng.uniform(-.15,.15,len(v)),v,s=4,color="k",alpha=.55,lw=0,zorder=3)
    ax.set_xticks(range(1,12)); ax.set_xticklabels([x.replace("All features","All") for x in METHODS],rotation=60,ha="right",fontsize=6); ax.set_title(lab,fontsize=7.5); ax.grid(axis="x",alpha=0)
fig.tight_layout(w_pad=0.8); fig.savefig(OUT+"rheum_box.pdf"); plt.close()
# 6 feature frequency
FR=pd.read_csv("featfreq.csv",index_col=0); M=FR.values
fig,ax=plt.subplots(figsize=(3.45,2.6)); im=ax.imshow(M,cmap="Blues",vmin=0,vmax=1,aspect="auto")
ax.set_xticks(range(14)); ax.set_xticklabels(FR.columns,rotation=60,ha="right",fontsize=6.2); ax.set_yticks(range(len(FR))); ax.set_yticklabels(FR.index,fontsize=6.5); ax.grid(False)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v=M[i,j]
        if v>0: ax.text(j,i,"1" if v==1 else f"{v:.1f}".lstrip("0"),ha="center",va="center",fontsize=4.8,color="white" if v>0.6 else "#333")
for sp in ax.spines.values(): sp.set_visible(False)
cb=fig.colorbar(im,ax=ax,fraction=0.04,pad=0.02); cb.ax.tick_params(labelsize=6); cb.set_label("Selection frequency (10 runs)",fontsize=6.5)
fig.savefig(OUT+"feat_freq.pdf"); plt.close()
# 7 trace
ARMC={"explore":"#c9c8c0","PERTURB":"#1baf7a","RESTART":"#e87ba4","DO":"#2a78d6","HO":"#eb6834"}
fig,axs=plt.subplots(2,2,figsize=(7.16,2.6),gridspec_kw=dict(height_ratios=[3,1]),sharex="col")
for k,(H,Hd,title,B) in enumerate([(rh[("Rheumatic","RL-DOHO",0)]["hist"],rh[("Rheumatic","DO",0)]["hist"],"Rheumatic, seed 0",252),(ge[("colon","RL-DOHO",0)]["hist"],ge[("colon","DO",0)]["hist"],"Colon, seed 0",1020)]):
    ax=axs[0,k]
    ax.step(H.evals,H.best,where="post",color=COL["RL-DOHO"],lw=1.8,label="RL-DOHO best")
    ax.step(Hd.evals,Hd.best,where="post",color=COL["DO"],lw=1.1,label="DO best (same seed)")
    ax.set_title(title); ax.set_ylabel("Fitness" if k==0 else "")
    if k==0: ax.legend(frameon=False,fontsize=6)
    ax2=axs[1,k]; ev=H.evals.values
    for i in range(1,len(H)):
        a=H.action.iloc[i].split("(")[0]; ax2.barh(0,ev[i]-ev[i-1],left=ev[i-1],height=1,color=ARMC.get(a,"#999"),lw=0)
    ax2.set_yticks([]); ax2.set_xlabel("Fitness evaluations"); ax2.grid(False); ax2.spines["left"].set_visible(False); ax2.set_xlim(0,B)
fig.legend(handles=[Patch(color=c,label=("DO exploration" if a=="explore" else a+" arm")) for a,c in ARMC.items()],ncol=5,frameon=False,loc="lower center",bbox_to_anchor=(0.5,-0.06))
fig.tight_layout(rect=(0,0.05,1,1),h_pad=0.3); fig.savefig(OUT+"trace.pdf"); plt.close()
# 8 ablation bars (same budget: rheum, genes; 3x budget: both)
A=load_abl("v2_ablation"); A["variant"]=A.variant.replace({"RL-DOHO v2 (UCB, gain reward)":"RL-DOHO (UCB, gain)","Dense start (as Notebooks 1-5)":"Dense start","No intervention (DO only)":"No intervention (DO)","UCB, old reward":"UCB, old reward"})
A["r"]=A.groupby(["dataset","seed"]).fit.transform(lambda x: rankdata(np.round(x,12)))
Lg=load_abl("v2_long"); Lg["variant"]=Lg.variant.replace({"UCB, gain reward":"RL-DOHO (UCB, gain)"}); Lg["r"]=Lg.groupby(["dataset","seed"]).fit.transform(lambda x: rankdata(np.round(x,12)))
order=["RL-DOHO (UCB, gain)","UCB, old reward","Random arm","PERTURB only","No tie-break","Dense start","No intervention (DO)"]
VC={"RL-DOHO (UCB, gain)":"#eda100","UCB, old reward":"#4a3aa7","Random arm":"#c9c8c0","PERTURB only":"#1baf7a","No tie-break":"#e87ba4","Dense start":"#8a8a8a","No intervention (DO)":"#2a78d6"}
groups=[("Rheumatic, same budget (10 runs)",A[A.dataset=="Rheumatic"]),("Gene data, same budget (50 runs)",A[A.dataset!="Rheumatic"]),("All data, 3$\\times$ budget (60 runs)",Lg)]
fig,axs=plt.subplots(1,3,figsize=(7.16,2.0),sharey=True)
for ax,(t,X) in zip(axs,groups):
    rk=X.groupby("variant").r.mean().reindex(order)
    for i,(v,x) in enumerate(rk.items()):
        y=len(order)-1-i
        if np.isnan(x): ax.text(0.1,y,"not run",va="center",fontsize=6,color="#888"); continue
        ax.barh(y,x,color=VC[v],height=0.65,edgecolor="white",lw=0.8); ax.text(x+0.05,y,f"{x:.2f}",va="center",fontsize=6)
    ax.set_title(t); ax.set_xlabel("Mean rank (lower is better)"); ax.grid(axis="y",alpha=0)
axs[0].set_yticks(range(len(order))[::-1]); axs[0].set_yticklabels(order)
fig.tight_layout(w_pad=0.6); fig.savefig(OUT+"ablation.pdf"); plt.close()
# 9 sensitivity: v1 vs v2 mean fitness per dataset for the family (dumbbell)
V1=load_v1(); S1=V1.groupby(["dataset","method"]).fit.mean(); S2=R.groupby(["dataset","method"]).fit.mean()
fig,axs=plt.subplots(1,6,figsize=(7.16,1.9))
for k,ds in enumerate(DS):
    ax=axs[k]
    for i,a in enumerate(FAMILY):
        y=len(FAMILY)-1-i; x1,x2=S1[(ds,a)],S2[(ds,a)]
        ax.plot([x1,x2],[y,y],color="#bbb",lw=1.2,zorder=1)
        ax.scatter(x1,y,s=16,facecolor="white",edgecolor=COL[a],lw=1,zorder=2); ax.scatter(x2,y,s=18,color=COL[a],zorder=3)
    ax.set_title(ds); ax.set_yticks(range(4)[::-1]); ax.set_yticklabels(FAMILY if k==0 else []); ax.tick_params(axis="x",labelsize=5.8)
    ax.margins(x=0.15)
fig.text(0.5,-0.04,"Mean fitness (hollow: simplified operators, dense start; filled: published operators, sparse start on gene data)",ha="center",fontsize=6.8)
fig.tight_layout(w_pad=0.4); fig.savefig(OUT+"sensitivity.pdf"); plt.close()
print("ok")
