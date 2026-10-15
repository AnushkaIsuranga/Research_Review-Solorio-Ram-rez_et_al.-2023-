import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5})
NAVY='#1F3A5F'; TEAL='#2A9D8F'; AMBER='#E9A23B'; RED='#C0392B'; GREY='#7F8C8D'; LIGHT='#EAF1F8'; GREEN='#2E7D32'
DPI=220

def box(ax,x,y,w,h,text,fc=LIGHT,ec=NAVY,ls='-',fs=8.3,tc='black',lw=1.3,bold=False):
    p=FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.02,rounding_size=0.12',fc=fc,ec=ec,ls=ls,lw=lw)
    ax.add_patch(p)
    ax.text(x,y,text,ha='center',va='center',fontsize=fs,color=tc,fontweight='bold' if bold else 'normal',linespacing=1.25)
def arrow(ax,x1,y1,x2,y2,c=NAVY,ls='-',lw=1.4):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=11,color=c,lw=lw,ls=ls,shrinkA=0,shrinkB=0))

# ---------- R1 author's workflow ----------
fig,ax=plt.subplots(figsize=(6.4,8.2)); ax.set_xlim(0,10); ax.set_ylim(0,16); ax.axis('off')
box(ax,5,15.0,8.6,1.2,'SDSS DR14 (10,000 rows)  |  DR16 (100,000)  |  DR17 (100,000)\nKaggle CSV files  [paper refs 31–33]',fc='#D6E4F0')
arrow(ax,5,14.4,5,13.6)
box(ax,5,12.9,8.6,1.4,'17 raw attributes + class label (STAR / GALAXY / QSO)\nNO EDA  ·  NO cleaning  ·  NO feature selection  ·  NO scaling',fc='#FDECEA',ec=RED)
arrow(ax,5,12.2,5,11.4)
box(ax,5,10.7,8.6,1.4,'Stratified hold-out split 80 % / 20 %  (single split)\nE ∩ P = ∅ ,  E ∪ P = D     (paper Eq. 5)')
arrow(ax,3.0,10.0,2.6,9.2); arrow(ax,7.0,10.0,7.4,9.2)
box(ax,2.6,8.2,4.6,1.9,'Claimed "core proposal":\nHyperparameter tuning by\nRandomized Search\n(no grid, no validation set,\nnot executed)',fc='white',ec=RED,ls='--',tc=RED,fs=7.8)
box(ax,7.4,8.2,4.6,1.9,'Baselines in WEKA\nNaïve Bayes · IB1 · IB3\nSMO (SVM) · MLP\n(settings not reported)',fs=8)
arrow(ax,2.6,7.25,2.6,6.45,c=RED,ls='--')
box(ax,2.6,5.6,4.6,1.7,'Random forest in Python /\nscikit-learn  (default settings)\n100 trees · max_depth = None\nm = √p · bootstrap · seed = 1',fc='#D6EFEA',ec=TEAL,fs=7.8)
arrow(ax,2.6,4.75,4.2,3.95); arrow(ax,7.4,7.25,7.4,4.6); arrow(ax,7.4,4.6,5.9,3.95)
box(ax,5,3.3,8.6,1.1,'Predictions on test set P  →  3 × 3 confusion matrix',fs=8.3)
arrow(ax,5,2.75,5,2.05)
box(ax,5,1.45,8.6,1.2,'Sensitivity · Specificity · "Balanced accuracy"\n(macro one-vs-rest averages, reported as "GALAXY positive")',fs=8)
ax.text(5,0.3,'Friedman test across 3 datasets (N = 3), p = 0.0262 — no post-hoc test',ha='center',fontsize=8,style='italic',color=NAVY)
arrow(ax,5,0.85,5,0.55)
fig.savefig('R1_author_workflow.png',dpi=DPI,bbox_inches='tight'); plt.close()

# ---------- R2 corrected pipeline ----------
fig,ax=plt.subplots(figsize=(6.4,9.0)); ax.set_xlim(0,10); ax.set_ylim(0,18); ax.axis('off')
y=17.2
box(ax,5,y,8.8,1.0,'Raw SDSS CSVs (DR14, DR16, DR17) + provenance log',fc='#D6E4F0'); arrow(ax,5,y-0.5,5,y-1.0)
y=15.7
box(ax,5,y,8.8,1.0,'1. Data audit & full EDA: moments, distribution tests, correlation, MI,\nduplicates, constant columns, sentinel values, cross-release overlap',fs=8); arrow(ax,5,y-0.5,5,y-1.0)
y=14.2
box(ax,5,y,8.8,1.0,'2. Cleaning: flag/remove −9999 sentinels · drop constant column (rerun_ID)\n· de-duplicate · float64-safe handling of 64-bit IDs',fs=8); arrow(ax,5,y-0.5,5,y-1.0)
y=12.7
box(ax,5,y,8.8,1.0,'3. Feature policy: DROP the 9 identifier / metadata columns',fc='#E8F5E9',ec=GREEN,bold=True,fs=8.3)
arrow(ax,3.0,y-0.5,2.6,y-1.15); arrow(ax,7.0,y-0.5,7.4,y-1.15)
y=11.0
box(ax,2.6,y,4.5,1.15,'Scenario S1 (spectro-assisted)\nu,g,r,i,z, redshift (+ α, δ ablation)\n+ colours u−g, g−r, r−i, i−z',fs=7.6,fc='#E8F5E9',ec=GREEN)
box(ax,7.4,y,4.5,1.15,'Scenario S2 (photometric-only)\nu,g,r,i,z + colours\n(no redshift: realistic use-case)',fs=7.6,fc='#E8F5E9',ec=GREEN)
arrow(ax,2.6,y-0.58,4.3,y-1.3); arrow(ax,7.4,y-0.58,5.7,y-1.3)
y=9.4
box(ax,5,y,8.8,1.0,'4. Stratified 80/20 split → FROZEN test set (used once)\n|  development set (80 %)',fs=8); arrow(ax,5,y-0.5,5,y-1.0)
y=7.3
box(ax,5,y,8.8,2.3,'5. Outer repeated stratified k-fold on the development set\nInside every fold — scikit-learn Pipeline (fit on TRAINING part only):\n  Transformer (Yeo–Johnson / quantile for skewed features;\n  StandardScaler for kNN, SVM, MLP, NB only)  →  Model\n  + inner RandomizedSearchCV (n_iter = 60, 5-fold, macro-F1)',fs=7.9,fc='#FFF4E0',ec=AMBER); arrow(ax,5,y-1.15,5,y-1.65)
y=5.0
box(ax,5,y,8.8,1.0,'6. Refit best configuration on the full development set\n→ single evaluation on the frozen test set',fs=8); arrow(ax,5,y-0.5,5,y-1.0)
y=3.5
box(ax,5,y,8.8,1.0,'7. Metrics: per-class P/R/F1 · macro-F1 · balanced accuracy (macro recall)\n· MCC · PR-AUC (QSO) · 95 % CIs · fit/predict runtime',fs=7.8); arrow(ax,5,y-0.5,5,y-1.0)
y=2.0
box(ax,5,y,8.8,1.0,'8. Statistical comparison: paired tests within a dataset ·\nFriedman + Nemenyi across datasets (with stated limits)',fs=8)
fig.savefig('R2_corrected_pipeline.png',dpi=DPI,bbox_inches='tight'); plt.close()

# ---------- R3 class distribution (computed from paper numbers) ----------
data={'DR14':(4998,850,4152),'DR16':(51323,10581,38096),'DR17':(59445,18961,21594)}
fig,ax=plt.subplots(figsize=(6.3,3.4)); w=0.26; cols=[NAVY,AMBER,TEAL]; labs=['GALAXY','QSO','STAR']
for j,(k,(g,q,s)) in enumerate(data.items()):
    n=g+q+s
    for i,v in enumerate((g,q,s)):
        ax.bar(j+(i-1)*w,100*v/n,w,color=cols[i],label=labs[i] if j==0 else None)
        ax.text(j+(i-1)*w,100*v/n+1,f'{v:,}\n({100*v/n:.1f}%)',ha='center',fontsize=6.6)
ax.set_xticks(range(3)); ax.set_xticklabels([f'SDSS {k}\n(n = {sum(v):,})\nIR = {int(v[0]/v[1]*100)/100:.2f}' for k,v in data.items()]); ax.set_ylim(0,75); ax.set_ylabel('Share of dataset (%)')
ax.legend(frameon=False,ncol=3,loc='upper right'); ax.spines[['top','right']].set_visible(False)
plt.subplots_adjust(bottom=0.2)
fig.savefig('R3_class_distribution.png',dpi=DPI,bbox_inches='tight'); plt.close()
for k,v in data.items():
    p=np.array(v)/sum(v); H=-(p*np.log(p)).sum()/np.log(3); print(k,'normalised entropy',round(H,4))

# ---------- R4 scale disparity (schematic) ----------
feat=[('obj_ID',18.1,'id'),('spec_obj_ID',18.6,'id'),('MJD',4.75,'id'),('plate',3.7,'id'),('run_ID',3.5,'id'),('rerun_ID',2.5,'id'),('field_ID',2.5,'id'),('fiber_ID',2.7,'id'),('cam_col',0.5,'id'),
      ('alpha',2.3,'pos'),('delta',1.4,'pos'),('u',1.3,'phot'),('g',1.3,'phot'),('r',1.3,'phot'),('i',1.3,'phot'),('z',1.3,'phot'),('redshift',-0.7,'spec')]
cm={'id':RED,'pos':AMBER,'phot':NAVY,'spec':TEAL}
fig,ax=plt.subplots(figsize=(6.3,4.2))
feat=sorted(feat,key=lambda t:t[1])
ax.barh([f[0] for f in feat],[f[1] for f in feat],color=[cm[f[2]] for f in feat])
ax.axvline(0,color='black',lw=0.8)
ax.set_xlabel('≈ log10 of typical absolute value  (illustrative order of magnitude, not computed from the data)')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=RED,label='Identifier / metadata (9)'),Patch(color=AMBER,label='Sky position (2)'),Patch(color=NAVY,label='Photometric (5)'),Patch(color=TEAL,label='Spectroscopic (1)')],frameon=False,loc='lower right',fontsize=7.5)
ax.spines[['top','right']].set_visible(False)
ax.annotate('≈ 18 orders of magnitude\nbetween redshift and obj_ID',xy=(18.1,16),xytext=(8.3,7.2),fontsize=8,color=RED,arrowprops=dict(arrowstyle='->',color=RED))
fig.savefig('R4_feature_scales.png',dpi=DPI,bbox_inches='tight'); plt.close()

# ---------- R5 redshift schematic ----------
x=np.linspace(0,4.5,1500)
star=np.exp(-0.5*(x/0.004)**2); star=star/star.max()*1.0
gal=np.exp(-0.5*((np.log(x+1e-9)-np.log(0.12))/0.75)**2)/(x+1e-3); gal=gal/gal.max()*0.55
qso=np.exp(-0.5*((np.log(x+1e-9)-np.log(1.4))/0.6)**2)/(x+1e-3); qso=qso/qso.max()*0.22
fig,ax=plt.subplots(figsize=(6.3,3.1))
for c,l,col in [(star,'STAR  (spike at z ≈ 0)',TEAL),(gal,'GALAXY  (z ≈ 0 – 0.6, right-skewed)',NAVY),(qso,'QSO  (broad, long tail to z > 4)',AMBER)]:
    ax.fill_between(x,c,alpha=0.35,color=col); ax.plot(x,c,color=col,lw=1.3,label=l)
ax.set_xlabel('redshift (schematic)'); ax.set_ylabel('relative density (schematic)'); ax.set_yticks([])
ax.legend(frameon=False); ax.spines[['top','right']].set_visible(False)
ax.set_title('Expected class-conditional shape of redshift — abstract sketch, not plotted from data',fontsize=8,color=GREY)
fig.savefig('R5_redshift_schematic.png',dpi=DPI,bbox_inches='tight'); plt.close()

# ---------- R6 leakage schematic ----------
fig,ax=plt.subplots(figsize=(6.4,4.4)); ax.set_xlim(0,10); ax.set_ylim(0,9); ax.axis('off')
box(ax,5,8.1,3.6,0.9,'plate ≤ t₁ ?',fs=8.5,fc='#FDECEA',ec=RED)
box(ax,2.6,6.0,3.4,0.9,'MJD ≤ t₂ ?',fs=8.5,fc='#FDECEA',ec=RED); box(ax,7.4,6.0,3.4,0.9,'fiber_ID ≤ t₃ ?',fs=8.5,fc='#FDECEA',ec=RED)
arrow(ax,4.0,7.65,2.9,6.5); arrow(ax,6.0,7.65,7.1,6.5)
for xx,t,c in [(1.3,'QSO',AMBER),(3.9,'GALAXY',NAVY),(6.1,'GALAXY',NAVY),(8.7,'STAR',TEAL)]:
    box(ax,xx,4.0,2.0,0.8,t,fs=8,fc='white',ec=c)
arrow(ax,2.2,5.55,1.4,4.4); arrow(ax,3.0,5.55,3.8,4.4); arrow(ax,7.0,5.55,6.2,4.4); arrow(ax,7.8,5.55,8.6,4.4)
ax.text(5,2.8,'t₁, t₂, t₃ are illustrative thresholds — no value is claimed',ha='center',fontsize=7.5,style='italic',color=GREY)
box(ax,5,1.4,9.0,1.9,'Keys record WHERE / WHEN / HOW an object was observed, not WHAT it is.\nA fully grown tree can isolate any record with splits on a unique key (training error → 0),\nand survey target-selection can make key ranges correlate with class —\nso hold-out accuracy may partly reflect survey design rather than astrophysics.',fs=7.6,fc='#FFF8E1',ec=AMBER)
fig.savefig('R6_leakage_schematic.png',dpi=DPI,bbox_inches='tight'); plt.close()

# ---------- R7 confusion matrices ----------
cms={'DR14 (Paper Fig. 4)':np.array([[984,9,6],[8,161,0],[0,0,832]]),'DR16 (Paper Fig. 6)':np.array([[10000,31,27],[121,2002,1],[16,0,7607]]),'DR17 (Paper Fig. 8)':np.array([[11718,119,34],[227,3594,0],[4,0,4304]])}
fig,axs=plt.subplots(1,3,figsize=(6.6,2.7))
for ax,(t,cmx) in zip(axs,cms.items()):
    norm=cmx/cmx.sum(1,keepdims=True)
    ax.imshow(norm,cmap='Blues',vmin=0,vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j,i,f'{cmx[i,j]:,}\n{100*norm[i,j]:.1f}%',ha='center',va='center',fontsize=6.2,color='white' if norm[i,j]>0.55 else 'black')
    ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(['GAL','QSO','STAR'],fontsize=7); ax.set_yticklabels(['GAL','QSO','STAR'],fontsize=7)
    ax.set_title(t,fontsize=7.5); ax.set_xlabel('predicted',fontsize=7)
axs[0].set_ylabel('true',fontsize=7)
plt.tight_layout()
fig.savefig('R7_confusion_matrices.png',dpi=DPI,bbox_inches='tight'); plt.close()

# ---------- R8 Table 3 vs Figure 7 ----------
algs=['NB','IB1','IB3','SMO','MLP','RF']
dr16=[.9746,.8879,.8858,.9554,.9843,.9861]; dr17=[.7219,.8621,.8505,.9474,.9738,.9815]
fig,axs=plt.subplots(1,2,figsize=(6.6,2.9),sharey=True)
for ax,vals,t,c in [(axs[0],dr16,'Paper Table 3 — DR16 balanced accuracy',NAVY),(axs[1],dr17,'Bars drawn in Paper Fig. 7 (caption: DR16)\n≈ Paper Table 4 / Fig. 9 (DR17)',RED)]:
    ax.bar(algs,vals,color=c); ax.set_ylim(0.6,1.0); ax.set_title(t,fontsize=7.6)
    for i,v in enumerate(vals): ax.text(i,v+0.005,f'{v:.3f}',ha='center',fontsize=6.4)
    ax.spines[['top','right']].set_visible(False)
axs[0].set_ylabel('balanced accuracy')
plt.tight_layout()
fig.savefig('R8_fig7_mismatch.png',dpi=DPI,bbox_inches='tight'); plt.close()

# ---------- R9 nested CV schematic ----------
fig,ax=plt.subplots(figsize=(6.4,3.6)); ax.set_xlim(0,12); ax.set_ylim(0,8); ax.axis('off')
ax.text(0.1,7.5,'Outer loop (performance estimate) — development set = 80 %',fontsize=8,fontweight='bold',color=NAVY)
for f in range(5):
    for k in range(5):
        ax.add_patch(Rectangle((0.1+k*1.6,6.2-f*0.55),1.5,0.45,fc=NAVY if k==f else '#D6E4F0',ec='white'))
ax.text(0.1,3.62,'dark blue = outer test fold; light blue = outer training folds',fontsize=7,color=GREY)
ax.add_patch(Rectangle((9.6,3.6),2.2,3.05,fc=AMBER,ec='white')); ax.text(10.7,5.1,'FROZEN\ntest set 20 %\n(used once)',ha='center',va='center',fontsize=7.5)
ax.text(0.1,3.05,'Inner loop (hyper-parameter search) — inside the training part of ONE outer fold',fontsize=8,fontweight='bold',color=TEAL)
for f in range(5):
    for k in range(5):
        ax.add_patch(Rectangle((0.1+k*1.6,2.0-f*0.4),1.5,0.33,fc=TEAL if k==f else '#D6EFEA',ec='white'))
ax.text(8.3,1.3,'teal = validation fold\nscaler / transformers fitted\non the inner-training folds only',fontsize=7,color=GREY,va='center')
fig.savefig('R9_nested_cv.png',dpi=DPI,bbox_inches='tight'); plt.close()
print('done')
