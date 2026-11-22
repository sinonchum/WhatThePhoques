"""For a FIXED truth partition, search for public-endpoint subsets U that reproduce every observed score."""
import numpy as np, csv, itertools, sys, json, collections
FAM={'G1':"01 02 04 12 14 17 18 23 24 25 33",'G2':"03 06 07 08 20 21 22 26 28 30 32",'G3':"05 09 10 11 13 15 16 19 27 29 31"}
eps=[f"ep-{i:02d}" for i in range(1,34)]; fam={}
for g,m in FAM.items():
    for x in m.split(): fam['ep-'+x]=g
allp=list(itertools.combinations(range(33),2)); A=np.array([a for a,b in allp]); B=np.array([b for a,b in allp])
SFm=np.array([fam[eps[a]]==fam[eps[b]] for a,b in allp]); sff=SFm.astype(float)
def load(path):
    F={};M={}
    for r in csv.reader(open(path)):
        if r and r[0].startswith('ep-'): F[r[0]]=r[1]; M[r[0]]=(r[1],r[2])
    return np.array([F[eps[a]]==F[eps[b]] for a,b in allp],dtype=float),np.array([M[eps[a]]==M[eps[b]] for a,b in allp],dtype=float)
OBS=json.load(open(sys.argv[1])); PF=np.stack([load(p)[0] for n,p,s,f in OBS]); PM=np.stack([load(p)[1] for n,p,s,f in OBS]); SC=np.array([s for n,p,s,f in OBS])
def mccv(pred,truth,w):
    tp=pred@(truth*w); pp=pred@w; fp=pp-tp; fn=(truth*w).sum()-tp; tn=w.sum()-tp-fp-fn
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); return np.maximum(np.where(den>0,(tp*tn-fp*fn)/np.maximum(den,1e-12),0.0),0)
def truth(groups):
    lab={e:e for e in eps}
    for gi,grp in enumerate(groups):
        for x in grp.split(): lab['ep-'+x]=f'grp{gi}'
    return np.array([SFm[k] and lab[eps[a]]==lab[eps[b]] for k,(a,b) in enumerate(allp)],dtype=float)
def miss(t,u):
    w=u[A]*u[B]; s=0.1*mccv(PF,sff,w)+0.9*mccv(PM,t,w*sff); return (np.abs(s-SC)-0.025+1e-9).clip(0)
def search(t,restarts,seed=0,sizes=(12,24)):
    found=[]; best=9
    for r in range(restarts):
        rr=np.random.default_rng(seed*1000+r); u=(rr.random(33)<rr.uniform(0.35,0.7)).astype(float); cur=miss(t,u).sum()
        for it in range(6000):
            if cur<1e-9: break
            i=rr.integers(33); u[i]=1-u[i]; c2=miss(t,u).sum()
            if c2<=cur or rr.random()<np.exp(-(c2-cur)/0.015): cur=c2
            else: u[i]=1-u[i]
        best=min(best,cur)
        if cur<1e-9: found.append(u.copy())
    return found,best
if __name__=='__main__':
    MAIN=["02 04 23 33","06 07 08 20 30 32","10 16 29 31"]
    H={'main only':MAIN,
       '+G1 7 plain':MAIN+["01 12 14 17 18 24 25"], '+G1 {12,14,18,24,25}':MAIN+["12 14 18 24 25"],
       '+G2 A={03,21,22}':MAIN+["03 21 22"], '+G3 core={09,13,15,19}':MAIN+["09 13 15 19"], '+G3 {05,11}':MAIN+["05 11"],
       '+all second groups':MAIN+["01 12 14 17 18 24 25","03 21 22","09 13 15 19","05 11"],
       '+A,+core,+{05,11}':MAIN+["03 21 22","09 13 15 19","05 11"],
       'G2 20 with A':["02 04 23 33","06 07 08 30 32","10 16 29 31","03 20 21 22"],
       'G3 27 in main':["02 04 23 33","06 07 08 20 30 32","10 16 27 29 31"],
       'G3 16,29 with core':["02 04 23 33","06 07 08 20 30 32","10 31","09 13 15 16 19 29"],
       'G1 main + 14':["02 04 14 23 33","06 07 08 20 30 32","10 16 29 31"]}
    R=int(sys.argv[2])
    for name,groups in H.items():
        t=truth(groups); found,best=search(t,R)
        if found:
            F=np.array(found); sz=F.sum(1)
            print(f"{name:26} consistent in {len(found):3d}/{R} restarts; |U| {sz.min():.0f}-{sz.max():.0f}; always public: {' '.join(eps[i][-2:] for i in range(33) if F[:,i].mean()>0.97)}; never: {' '.join(eps[i][-2:] for i in range(33) if F[:,i].mean()<0.03)}")
        else: print(f"{name:26} NO consistent public set in {R} restarts (best residual {best:.3f})")
