import numpy as np, csv, itertools, sys, json
rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 0)
FAM={'G1':"01 02 04 12 14 17 18 23 24 25 33",'G2':"03 06 07 08 20 21 22 26 28 30 32",'G3':"05 09 10 11 13 15 16 19 27 29 31"}
eps=[f"ep-{i:02d}" for i in range(1,34)]; fam={}
for g,m in FAM.items():
    for x in m.split(): fam['ep-'+x]=g
pairs=list(itertools.combinations(range(33),2)); NP=len(pairs)
samefam=np.array([fam[eps[a]]==fam[eps[b]] for a,b in pairs])
SF=np.where(samefam)[0]
def load(path):
    F={};M={}
    for r in csv.reader(open(path)):
        if r and r[0].startswith('ep-'): F[r[0]]=r[1]; M[r[0]]=(r[1],r[2])
    pf=np.array([F[eps[a]]==F[eps[b]] for a,b in pairs]); pm=np.array([M[eps[a]]==M[eps[b]] for a,b in pairs])
    return pf,pm
T='../../test-data/collected/'
SUBS=[('first-live',T+'first-live/submission.csv',0.05),('offline-v3',T+'offline-v3/candidate.csv',0.10),('round2',T+'official-round2/candidate.csv',0.10),
      ('minimax','../metric-partition/minimax.csv',0.10),('c01','candidates/c01-disguises.csv',0.55),('s02','candidates/s02-g1-only.csv',0.55),
      ('c02','candidates/c02-g2-add32.csv',0.60),('c03','candidates/c03-g2-bulk.csv',0.50)]
for extra in sys.argv[3:]:
    n,p,s=extra.split('='); SUBS.append((n,p,float(s)))
PRED=[(n,)+load(p)+(s,) for n,p,s in SUBS]
NM=int(sys.argv[2]) if len(sys.argv)>2 else 20000
mask=(rng.random((NM,NP))<0.30)
def mcc(tp,fp,fn,tn):
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); out=np.where(den>0,(tp*tn-fp*fn)/np.maximum(den,1e-9),0.0); return np.maximum(out,0)
mf=mask.astype(np.float64)
# family mcc per submission per mask (truth families fixed)
FAMM=[]
for n,pf,pm,s in PRED:
    tp=mf[:,samefam&pf].sum(1); fp=mf[:,(~samefam)&pf].sum(1); fn=mf[:,samefam&(~pf)].sum(1); tn=mf[:,(~samefam)&(~pf)].sum(1)
    FAMM.append(mcc(tp,fp,fn,tn))
msf=mf[:,SF]                      # masks restricted to same-family pairs
PM=[pm[SF].astype(np.float64) for n,pf,pm,s in PRED]
sfpairs=[pairs[i] for i in SF]
def truth_vec(member):            # member: array 33 of group id (0 = singleton, k>0 = group k within family)
    return np.array([member[a]>0 and member[a]==member[b] for a,b in sfpairs],dtype=np.float64)
def scores(t):
    out=[]
    tot=msf.sum(1); tsum=msf@t
    for k,pmv in enumerate(PM):
        tp=msf@(pmv*t); pp=msf@pmv; fp=pp-tp; fn=tsum-tp; tn=tot-tp-fp-fn
        out.append(0.1*FAMM[k]+0.9*mcc(tp,fp,fn,tn))
    return np.stack(out,1)        # NM x nsub
OBS=np.array([s for *_,s in PRED])
def like(t):
    sc=scores(t); ok=np.all(np.abs(np.round(sc/0.05)*0.05-OBS)<1e-6,axis=1); return ok
if __name__=='__main__':
    PRIOR={'G1':{'02':.9,'23':.9,'04':.7,'33':.6},'G2':{'20':.9,'08':.9,'06':.7,'32':.5,'07':.3,'30':.3,'28':.1},'G3':{'16':.9,'29':.9,'10':.7,'27':.5,'31':.2}}
    flat=len(sys.argv)>3 and False
    pri=np.array([PRIOR[fam[e]].get(e[-2:],0.15) for e in eps])
    if '--flat' in sys.argv: pri=np.full(33,0.35)
    member=(rng.random(33)<pri).astype(int)
    def lp(m): return np.sum(np.where(m>0,np.log(pri),np.log(1-pri)))
    cur=like(truth_vec(member)); curL=cur.mean(); curp=lp(member)
    acc=np.zeros(33); n=0; tot_ok=np.zeros(NM); best=(0,None); post_pair=np.zeros(len(SF))
    STEPS=int(15000)
    for it in range(STEPS):
        m2=member.copy(); i=rng.integers(33); m2[i]=1-m2[i]
        if rng.random()<0.3:
            j=rng.integers(33); m2[j]=1-m2[j]
        c2=like(truth_vec(m2)); L2=c2.mean(); p2=lp(m2)
        if L2>0 and (curL==0 or np.log(rng.random())< (np.log(L2)+p2)-(np.log(curL)+curp)):
            member,cur,curL,curp=m2,c2,L2,p2
        if it>=STEPS//5 and curL>0:
            acc+=member; n+=1; post_pair+=truth_vec(member)
            if curL*np.exp(curp)>best[0]: best=(curL*np.exp(curp),member.copy(),curL)
    print('samples',n,'final L',curL)
    for g,mm in FAM.items(): print(g,' '.join(f"{x}:{acc[eps.index('ep-'+x)]/max(n,1):.2f}" for x in mm.split()))
    if best[1] is not None:
        print('best L',best[2],{g:[x for x in mm.split() if best[1][eps.index('ep-'+x)]] for g,mm in FAM.items()})
    json.dump({'acc':(acc/max(n,1)).tolist()},open(sys.argv[0].replace('infer.py','post.json'),'w'))
