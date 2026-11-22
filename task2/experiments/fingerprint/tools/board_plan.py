import sys,os,numpy as np,collections,json,itertools
HERE=os.path.dirname(os.path.abspath(__file__))
args=[a for a in sys.argv[1:]]
seed=args[0]; NMs=args[1]; STEPS=int(args[2]); extra=[a for a in args[3:] if '=' in a]
sys.argv=[sys.argv[0],seed,NMs]+extra
import importlib.util
spec=importlib.util.spec_from_file_location('infer',os.path.join(HERE,'board_infer.py')); I=importlib.util.module_from_spec(spec); spec.loader.exec_module(I)
np.seterr(all='ignore')
rng=I.rng; eps=I.eps; fam=I.fam; FAM=I.FAM; sfpairs=I.sfpairs
# prior: P(label=1 main group), P(label=2 second group), else singleton
P1={'G1':{'02':.9,'23':.9,'04':.7,'33':.6},'G2':{'20':.7,'08':.9,'06':.7,'32':.5,'07':.4,'30':.4,'28':.08},'G3':{'16':.9,'29':.9,'10':.7,'27':.4,'31':.2}}
P2={'G1':{'01':.2,'12':.25,'14':.25,'17':.15,'18':.25,'24':.25,'25':.25,'33':.1},'G2':{'03':.3,'21':.3,'22':.3,'26':.2,'20':.2,'28':.08},'G3':{'05':.3,'11':.3,'09':.25,'13':.25,'15':.25,'19':.25,'27':.15,'31':.15}}
p1=np.array([P1[fam[e]].get(e[-2:],0.12) for e in eps]); p2=np.array([P2[fam[e]].get(e[-2:],0.05) for e in eps]); p0=1-p1-p2
assert (p0>0).all()
LP=np.log(np.stack([p0,p1,p2],1))
lp=lambda m: LP[np.arange(33),m].sum()
base=json.loads(open(os.path.join(HERE,'board_base.json')).read()); BEST=float(open(os.path.join(HERE,'board_best.txt')).read())
member=np.array([1 if e[-2:] in sum(base.values(),[]) else 0 for e in eps])
cur=I.like(I.truth_vec(member)); curL=cur.mean(); curp=lp(member)
print('start L',curL,flush=True)
samples=[]; cnt=np.zeros((33,3)); n=0; pairpost=np.zeros(len(sfpairs))
for it in range(STEPS):
    m2=member.copy(); i=rng.integers(33); m2[i]=(m2[i]+rng.integers(1,3))%3
    if rng.random()<0.3:
        j=rng.integers(33); m2[j]=(m2[j]+rng.integers(1,3))%3
    c2=I.like(I.truth_vec(m2)); L2=c2.mean(); q=lp(m2)
    if L2>0 and (curL==0 or np.log(rng.random())<(np.log(L2)+q)-(np.log(curL)+curp)):
        member,cur,curL,curp=m2,c2,L2,q
    if it>=STEPS//5 and curL>0:
        cnt[np.arange(33),member]+=1; n+=1; pairpost+=I.truth_vec(member)
        if it%8==0: samples.append((member.copy(),rng.choice(np.where(cur)[0])))
print('n',n,'samples',len(samples))
for g,mm in FAM.items():
    print(g,'main  ',' '.join(f"{x}:{cnt[eps.index('ep-'+x),1]/n:.2f}" for x in mm.split()))
    print(g,'second',' '.join(f"{x}:{cnt[eps.index('ep-'+x),2]/n:.2f}" for x in mm.split()))
pp=pairpost/n
print('pairs with posterior same-model prob >= 0.35:')
print('  '+'  '.join(f"{eps[a][-2:]}-{eps[b][-2:]}:{pp[k]:.2f}" for k,(a,b) in enumerate(sfpairs) if pp[k]>=0.35))
def predvec(groups):   # groups: dict fam -> list of lists
    lab={}
    for g,mm in FAM.items():
        for x in mm.split(): lab['ep-'+x]=g+x
        for gi,grp in enumerate(groups.get(g,[])):
            for x in grp: lab['ep-'+x]=f'{g}grp{gi}'
    return np.array([lab[eps[a]]==lab[eps[b]] for a,b in sfpairs],dtype=np.float64)
def evalc(groups):
    pv=predvec(groups); pub=[];hid=[]
    for m,k in samples:
        t=I.truth_vec(m)
        for sel,out in ((I.msf[k]>0,pub),(I.msf[k]==0,hid)):
            tp=(pv*t)[sel].sum(); fp=(pv*(1-t))[sel].sum(); fn=((1-pv)*t)[sel].sum(); tn=sel.sum()-tp-fp-fn
            den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); out.append(0.1+0.9*max(0,(tp*tn-fp*fn)/den if den>0 else 0))
    return np.round(np.array(pub)/0.05)*0.05,np.array(hid)
B={g:[list(v)] for g,v in base.items()}
cands={'current-best':B}
def mod(name,g,add=(),rem=(),second=None):
    d={k:[list(x) for x in v] for k,v in B.items()}; d[g][0]=[x for x in d[g][0] if x not in rem]+list(add)
    if second: d[g].append(list(second))
    cands[name]=d
for g,mm in FAM.items():
    for x in mm.split():
        if x in B[g][0]: mod(f'{g}-{x}',g,rem=[x])
        else: mod(f'{g}+{x}',g,add=[x])
mod('G2: A2={03,21,22}','G2',second=['03','21','22']); mod('G2: A2={03,21,22,26}','G2',second=['03','21','22','26'])
mod('G2: 20->A2','G2',rem=['20'],second=['03','20','21','22'])
mod('G1: P2={12,14,18,24,25}','G1',second=['12','14','18','24','25']); mod('G1: P2={12,14,18,24,25,01,17}','G1',second=['12','14','18','24','25','01','17'])
mod('G1: P2={12,18}','G1',second=['12','18']); mod('G1: P2={24,25}','G1',second=['24','25'])
mod('G3: C2={09,13,15,19}','G3',second=['09','13','15','19']); mod('G3: main+{09,13,15,19}','G3',add=['09','13','15','19'])
mod('G3: N2={05,11}','G3',second=['05','11']); mod('G3: D2={27,31}','G3',second=['27','31'])
mod('G3: main+27 & {05,11}','G3',add=['27'],second=['05','11'])
print(f"{'candidate':30} E[pub] P(>best) P(<best) E[hidden]  entropy   pub distribution")
rows=[]
for name,gr in cands.items():
    pub,hid=evalc(gr); c=collections.Counter(np.round(pub,2)); pr=np.array(list(c.values()))/len(pub); H=-(pr*np.log2(pr)).sum()
    rows.append((hid.mean(),name,pub.mean(),(pub>BEST+0.001).mean(),(pub<BEST-0.001).mean(),H,dict(sorted(c.items()))))
for h,name,pm,pg,pl,H,c in sorted(rows,reverse=True):
    print(f"{name:30} {pm:.3f}  {pg:.2f}     {pl:.2f}     {h:.3f}     {H:.2f}    "+' '.join(f"{k}:{v}" for k,v in c.items()))
