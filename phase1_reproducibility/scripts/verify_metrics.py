import numpy as np
from scipy import stats
cms = {
 'DR14': np.array([[984,9,6],[8,161,0],[0,0,832]]),
 'DR16': np.array([[10000,31,27],[121,2002,1],[16,0,7607]]),
 'DR17': np.array([[11718,119,34],[227,3594,0],[4,0,4304]]),
}
full = {'DR14':(4998,850,4152),'DR16':(51323,10581,38096),'DR17':(59445,18961,21594)}
for k,cm in cms.items():
    n=cm.sum(); rows=cm.sum(1); cols=cm.sum(0)
    print('\n',k,'N=',n,'rows',rows,'cols',cols)
    exp = np.array(full[k])*0.2
    print(' expected 20% per class (G,Q,S):',exp, 'total expected',sum(full[k])*0.2)
    acc=np.trace(cm)/n
    # galaxy positive
    TP=cm[0,0];FN=rows[0]-TP;FP=cols[0]-TP;TN=n-TP-FN-FP
    se=TP/(TP+FN);sp=TN/(TN+FP)
    print(' acc %.4f | GAL-pos sens %.4f spec %.4f BA %.4f'%(acc,se,sp,(se+sp)/2))
    rec=np.diag(cm)/rows; prec=np.diag(cm)/cols
    specs=[]
    for i in range(3):
        tp=cm[i,i];fn=rows[i]-tp;fp=cols[i]-tp;tn=n-tp-fn-fp;specs.append(tn/(tn+fp))
    print(' per-class recall',rec.round(4),'prec',prec.round(4),'spec',np.round(specs,4))
    print(' macro sens %.4f macro spec %.4f -> (avg) %.4f | macro-recall BA %.4f | macroF1 %.4f'%(rec.mean(),np.mean(specs),(rec.mean()+np.mean(specs))/2,rec.mean(),np.mean(2*prec*rec/(prec+rec))))
    # MCC multiclass
    t=rows; p=cols; c=np.trace(cm); s=n
    mcc=(c*s-np.dot(t,p))/np.sqrt((s**2-np.dot(p,p))*(s**2-np.dot(t,t)))
    po=acc; pe=np.dot(rows,cols)/n**2; kappa=(po-pe)/(1-pe)
    print(' MCC %.4f kappa %.4f'%(mcc,kappa))
    # Wilson CI acc
    z=1.96; ph=acc; den=1+z*z/n; centre=(ph+z*z/(2*n))/den; half=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
    print(' acc 95%% Wilson CI [%.4f,%.4f]'%(centre-half,centre+half))
    # QSO recall CI
    q=rows[1]; r=rec[1]; centre=(r+z*z/(2*q))/(1+z*z/q); half=z*np.sqrt(r*(1-r)/q+z*z/(4*q*q))/(1+z*z/q)
    print(' QSO recall 95%% Wilson CI [%.4f,%.4f] n_qso=%d'%(centre-half,centre+half,q))
# Friedman
algs=['NB','IB1','IB3','SMO','MLP','RF']
BA={'DR14':[.9547,.8916,.8858,.9256,.9777,.9866],'DR16':[.9746,.8879,.8858,.9554,.9843,.9861],'DR17':[.7219,.8621,.8505,.9474,.9738,.9815]}
SE={'DR14':[.9409,.8581,.8466,.9080,.9710,.9792],'DR16':[.9630,.8515,.8544,.9433,.9788,.9783],'DR17':[.6329,.8185,.8022,.9313,.9657,.9756]}
SP={'DR14':[.9685,.9251,.9250,.9433,.9845,.9940],'DR16':[.9862,.9243,.9261,.9676,.9898,.9940],'DR17':[.8108,.9057,.8988,.9636,.9818,.9874]}
for name,T in [('BA',BA),('SENS',SE),('SPEC',SP)]:
    M=np.array([T[k] for k in ['DR14','DR16','DR17']])
    ranks=np.array([stats.rankdata(-r) for r in M])
    print(name,'mean ranks (best=1):',dict(zip(algs,ranks.mean(0).round(3))))
    print('  7-rank:',dict(zip(algs,(7-ranks.mean(0)).round(3))))
    print('  friedman',stats.friedmanchisquare(*M.T))
# check BA = (sens+spec)/2 in tables
for k in BA:
    print(k,[round((a+b)/2,4) for a,b in zip(SE[k],SP[k])], BA[k])
# combinatorics
from math import comb, sqrt
print('C(17,4)',comb(17,4),'P(no metadata in 4 of 17)',comb(8,4)/comb(17,4),'P(at least one metadata)',1-comb(8,4)/comb(17,4),'P(at least one physical)',1-comb(9,4)/comb(17,4))
print('sqrt17',sqrt(17),'sqrt8',sqrt(8))
print('bootstrap unique frac n=8000',1-(1-1/8000)**8000)
print('float64 spacing at 1.237e18', np.spacing(np.float64(1.237e18)))
