import numpy as np
from scipy import stats
algs=['NB','IB1','IB3','SMO','MLP','RF']
# Balanced accuracy columns of paper Tables 2-4 (rows = DR14, DR16, DR17)
printed=[[.9547,.8916,.8858,.9256,.9777,.9866],[.9746,.8879,.8858,.9554,.9843,.9861],[.7219,.8621,.8505,.9474,.9738,.9815]]
corrected=[r[:] for r in printed]; corrected[1][2]=round((0.8544+0.9261)/2,4)   # DR16 IB3 from its own sens/spec = 0.8903
for name,M in [('as printed (reproduces paper p=0.0262)',printed),('DR16 IB3 corrected to 0.8903',corrected)]:
    M=np.array(M); r=np.array([stats.rankdata(-x) for x in M])
    f=stats.friedmanchisquare(*M.T)
    print(name,'| chi2=%.3f p=%.4f'%(f.statistic,f.pvalue))
    print('  mean rank (1=best):',dict(zip(algs,r.mean(0).round(3))))
    print('  7 - rank (higher=better):',dict(zip(algs,(7-r.mean(0)).round(3))))
k,N,q=6,3,2.850
print('Nemenyi CD (alpha=.05, k=6, N=3) = %.3f'%(q*np.sqrt(k*(k+1)/(6*N))))
