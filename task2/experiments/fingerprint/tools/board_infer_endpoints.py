"""Like joint.py, but the public split is a subset of ENDPOINTS: a pair is public iff both endpoints are public."""
import numpy as np, csv, itertools, sys, json, collections
rng=np.random.default_rng(int(sys.argv[1])); STEPS=int(sys.argv[2])
FAM={'G1':"01 02 04 12 14 17 18 23 24 25 33",'G2':"03 06 07 08 20 21 22 26 28 30 32",'G3':"05 09 10 11 13 15 16 19 27 29 31"}
eps=[f"ep-{i:02d}" for i in range(1,34)]; fam={}
for g,m in FAM.items():
    for x in m.split(): fam['ep-'+x]=g
allp=list(itertools.combinations(range(33),2)); A=np.array([a for a,b in allp]); B=np.array([b for a,b in allp])
SFm=np.array([fam[eps[a]]==fam[eps[b]] for a,b in allp])
def load(path):
    F={};M={}
    for r in csv.reader(open(path)):
        if r and r[0].startswith('ep-'): F[r[0]]=r[1]; M[r[0]]=(r[1],r[2])
    return np.array([F[eps[a]]==F[eps[b]] for a,b in allp],dtype=float),np.array([M[eps[a]]==M[eps[b]] for a,b in allp],dtype=float)
OBS=json.load(open(sys.argv[3])); PF=np.stack([load(p)[0] for n,p,s,f in OBS]); PM=np.stack([load(p)[1] for n,p,s,f in OBS]); SC=np.array([s for n,p,s,f in OBS])
P1=json.load(open(sys.argv[4])); p1=np.array([P1['main'].get(e[-2:],P1['main_default']) for e in eps]); p2=np.array([P1['second'].get(e[-2:],P1['second_default']) for e in eps]); LP=np.log(np.stack([1-p1-p2,p1,p2],1))
PU=float(sys.argv[7]) if len(sys.argv)>7 else 0.55
sff=SFm.astype(float)
def tvec(m): return (SFm&(m[A]>0)&(m[A]==m[B])).astype(float)
def mccv(pred,truth,w):
    tp=pred@(truth*w); pp=pred@w; fp=pp-tp; fn=(truth*w).sum()-tp; tn=w.sum()-tp-fp-fn
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); return np.maximum(np.where(den>0,(tp*tn-fp*fn)/np.maximum(den,1e-12),0.0),0)
def scores(t,u,pf=None,pm=None):
    pf=PF if pf is None else pf; pm=PM if pm is None else pm
    w=u[A]*u[B]; return 0.1*mccv(pf,sff,w)+0.9*mccv(pm,t,w*sff)
def missv(t,u): return (np.abs(scores(t,u)-SC)-0.025+1e-9).clip(0)
base=json.load(open(sys.argv[5]))
def anneal(seed):
    r=np.random.default_rng(seed); m=np.array([1 if e[-2:] in sum(base.values(),[]) else 0 for e in eps]); u=(r.random(33)<PU).astype(float)
    t=tvec(m); cur=missv(t,u).sum()
    for it in range(40000):
        if cur<1e-9: break
        T=0.02
        if r.random()<0.6:
            i=r.integers(33); u[i]=1-u[i]; c2=missv(t,u).sum()
            if c2<=cur or r.random()<np.exp(-(c2-cur)/T): cur=c2
            else: u[i]=1-u[i]
        else:
            i=r.integers(33); old=m[i]; m[i]=(m[i]+r.integers(1,3))%3; t2=tvec(m); c2=missv(t2,u).sum()
            if c2<=cur or r.random()<np.exp(-(c2-cur)/T): cur=c2; t=t2
            else: m[i]=old
    return m,u,cur
chains=[]; NCH=int(sys.argv[6]) if len(sys.argv)>6 else 8; fails=0
lq=np.log(PU/(1-PU))
for c in range(NCH):
    m,u,cur=anneal(7000+c+int(sys.argv[1])*100)
    if cur>1e-9: fails+=1; continue
    t=tvec(m); lp=LP[np.arange(33),m].sum(); cnt=np.zeros((33,3)); ucnt=np.zeros(33); n=0; samples=[]
    for it in range(STEPS):
        if rng.random()<0.5:
            i=rng.integers(33); u[i]=1-u[i]
            if missv(t,u).sum()<1e-9 and np.log(rng.random())<(lq if u[i]==1 else -lq): pass
            else: u[i]=1-u[i]
        else:
            i=rng.integers(33); old=m[i]; m[i]=(m[i]+rng.integers(1,3))%3; t2=tvec(m); lp2=LP[np.arange(33),m].sum()
            if missv(t2,u).sum()<1e-9 and np.log(rng.random())<lp2-lp: t=t2; lp=lp2
            else: m[i]=old
        if it>=STEPS//5:
            cnt[np.arange(33),m]+=1; ucnt+=u; n+=1
            if it%100==0: samples.append((m.copy(),u.copy()))
    chains.append((cnt/n,ucnt/n,samples))
print('chains ok',len(chains),'failed to find a consistent start',fails)
if not chains: sys.exit()
post=np.mean([c for c,_,_ in chains],0); up=np.mean([x for _,x,_ in chains],0); usd=np.std([x for _,x,_ in chains],0)
for g,mm in FAM.items():
    print(g,'public',' '.join(f"{x}:{up[eps.index('ep-'+x)]:.2f}" for x in mm.split()))
    print(g,'main  ',' '.join(f"{x}:{post[eps.index('ep-'+x),1]:.2f}" for x in mm.split()))
    print(g,'second',' '.join(f"{x}:{post[eps.index('ep-'+x),2]:.2f}" for x in mm.split()))
samples=[s for *_,ss in chains for s in ss]; print('public endpoints per sample: mean %.1f'%np.mean([u.sum() for m,u in samples]))
def predvec(groups):
    lab={}
    for g,mm in FAM.items():
        for x in mm.split(): lab['ep-'+x]=g+x
        for gi,grp in enumerate(groups.get(g,[])):
            for x in grp: lab['ep-'+x]=f'{g}grp{gi}'
    return np.array([lab[eps[a]]==lab[eps[b]] for a,b in allp],dtype=float)
def evalc(groups):
    pv=predvec(groups)[None,:]; pub=[];hid=[]
    for m,u in samples:
        t=tvec(m); w=u[A]*u[B]; pub.append(0.1+0.9*mccv(pv,t,w*sff)[0]); hid.append(0.1+0.9*mccv(pv,t,(1-w)*sff)[0])
    return np.round(np.array(pub)/0.05)*0.05,np.array(hid)
BEST=max(SC); B0={g:[list(v)] for g,v in base.items()}; cands={'current-best':B0}
def mod(name,g,add=(),rem=(),second=None):
    d={k:[list(x) for x in v] for k,v in B0.items()}; d[g][0]=[x for x in d[g][0] if x not in rem]+list(add)
    if second: d[g].append(list(second))
    cands[name]=d
for g,mm in FAM.items():
    for x in mm.split():
        if x in B0[g][0]: mod(f'{g}-{x}',g,rem=[x])
        else: mod(f'{g}+{x}',g,add=[x])
mod('G2: A2={03,21,22}','G2',second=['03','21','22']); mod('G2: 20->A2','G2',rem=['20'],second=['03','20','21','22'])
mod('G1: P2={12,18}','G1',second=['12','18']); mod('G3: C2={09,13,15,19}','G3',second=['09','13','15','19']); mod('G3: N2={05,11}','G3',second=['05','11'])
mod('G1: P2=7 plain','G1',second=['01','12','14','17','18','24','25']); mod('G1: P2={12,14,18,24,25}','G1',second=['12','14','18','24','25'])
cands['ALL second groups']={'G1':[B0['G1'][0],['01','12','14','17','18','24','25']],'G2':[B0['G2'][0],['03','21','22']],'G3':[B0['G3'][0],['09','13','15','19'],['05','11']]}
cands['G2 A2 + G3 C2']={'G1':[B0['G1'][0]],'G2':[B0['G2'][0],['03','21','22']],'G3':[B0['G3'][0],['09','13','15','19']]}
mod('G3: C2+N2','G3',second=['09','13','15','19']); cands['G3: C2+N2']['G3'].append(['05','11'])
rows=[]
for name,gr in cands.items():
    pub,hid=evalc(gr); c=collections.Counter(np.round(pub,2)); pr=np.array(list(c.values()))/len(pub); H=-(pr*np.log2(pr)).sum()
    rows.append((hid.mean(),name,pub.mean(),(pub>BEST+0.001).mean(),(pub<BEST-0.001).mean(),H,dict(sorted(c.items()))))
print(f"{'candidate':26} E[pub] P(>best) P(<best) E[hidden] entropy  pub distribution   ({len(samples)} samples)")
for h,name,pm,pg,pl,H,c in sorted(rows,reverse=True)[:int(sys.argv[8]) if len(sys.argv)>8 else 26]:
    print(f"{name:26} {pm:.3f}  {pg:.2f}     {pl:.2f}     {h:.3f}    {H:.2f}   "+' '.join(f"{k}:{v}" for k,v in c.items()))
