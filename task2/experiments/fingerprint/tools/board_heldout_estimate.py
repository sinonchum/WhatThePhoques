import numpy as np,glob,itertools,sys
S=sys.argv[1]; rng=np.random.default_rng(0)
U=np.unique(np.concatenate([np.load(f) for f in glob.glob(S+'/masks_*.npy')]),axis=0)
FAM={'G1':"01 02 04 12 14 17 18 23 24 25 33",'G2':"03 06 07 08 20 21 22 26 28 30 32",'G3':"05 09 10 11 13 15 16 19 27 29 31"}
fam={x:g for g,m in FAM.items() for x in m.split()}
E=sorted(fam); ix={e:i for i,e in enumerate(E)}
pairs=[(a,b) for a,b in itertools.combinations(E,2) if fam[a]==fam[b]]
BASE={'G1':"02 04 23 33".split(),'G2':"06 07 08 20 30 32".split(),'G3':"10 13 16 27 29 31".split()}
PD={'02':.92,'04':.85,'23':.92,'33':.85,'06':.75,'20':.85,'07':.8,'30':.8,'08':.95,'32':.85,'10':.75,'16':.92,'29':.92,'31':.85,'13':.9,'27':.45}
def sample(u, p66):
    pub={e for e in E if u[int(e)-1]>0}; mem=set()
    for g in BASE:
        for e in BASE[g]:
            if e in pub or rng.random()<PD[e]: mem.add(e)
    hp=[e for e in "01 12 14 17 18 24 25".split() if e not in pub]
    r=rng.random(); k=2 if r<p66 else (1 if r<p66+(1-p66)*0.8 else 0)
    w=np.array([1.5 if e=='17' else 1.0 for e in hp]); 
    for e in rng.choice(hp,size=min(k,len(hp)),replace=False,p=w/w.sum()): mem.add(e)
    hp2=[e for e in "03 21 22".split() if e not in pub]
    if rng.random()<0.35: mem.add(rng.choice(hp2))
    for e in "26 28 05 09 11 19".split():
        if e not in pub and rng.random()<0.06: mem.add(e)
    return pub,mem
def mcc(pred,truth,w):
    tp=(pred&truth&w).sum(); fp=(pred&~truth&w).sum(); fn=(~pred&truth&w).sum(); tn=(~pred&~truth&w).sum()
    d=np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))); return max(0,(tp*tn-fp*fn)/d) if d>0 else 0
def vec(mem): return np.array([a in mem and b in mem for a,b in pairs])
C={'t14 (none)':[], 't11 (+12)':['12'], '+17':['17'], '+12+17':['12','17'], '+12+14':['12','14'], '+12+14+17':['12','14','17'],'+21':['21'],'+12+21':['12','21'],
   '-27':['-27'],'-06':['-06'],'-20':['-20'],'-10':['-10'],'+12 -27':['12','-27'],'+03+21+22':['03','21','22'],'+5 G1 plain hidden-ish (01 12 14 17 18)':['01','12','14','17','18']}
allm=set(sum(BASE.values(),[]))
for p66 in (0.3,0.5,0.7):
    N=4000; res={k:[] for k in C}
    for n in range(N):
        u=U[rng.integers(len(U))]; pub,mem=sample(u,p66); t=vec(mem); w=np.array([not(a in pub and b in pub) for a,b in pairs])
        for k,mods in C.items():
            m=set(allm)
            for x in mods:
                if x.startswith('-'): m.discard(x[1:])
                else: m.add(x)
            res[k].append(0.1+0.9*mcc(vec(m),t,w))
    print('P(6+5 in G1)=',p66)
    for k,v in sorted(res.items(),key=lambda kv:-np.mean(kv[1])): print(f"   {k:42} mean {np.mean(v):.3f}  p10 {np.percentile(v,10):.2f}  p90 {np.percentile(v,90):.2f}")
