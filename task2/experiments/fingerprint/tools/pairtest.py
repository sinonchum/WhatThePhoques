"""Pairwise permutation test between endpoints of one family (statistic from the AgentProv paper).

  python3 tools/pairtest.py G3 n1,n2,n3,n4 data/claude-code-run/*.jsonl
  python3 tools/pairtest.py refs n1,n2 data/claude-code-run/*.jsonl

First argument: G1, G2, G3 or refs.  Second: comma-separated prompt ids (numeric ids n1-n4, or t1).
Remaining arguments: reply files (.jsonl) or bridge result directories containing answers.jsonl.
"""
import collections,json,random,sys,itertools,re,os
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import analyze_num as A
OPT=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','t1_options.json')))
def norm(v): return v.split('.')[0] if v.replace('.','',1).lstrip('-').isdigit() else v
def parse_t1(text):
    out={}
    for m in re.finditer(r'^\s*(\d{1,2})\s*[.):]\s*(.*)$',text or '',re.M):
        i=int(m.group(1))-1
        if 0<=i<16 and i not in out:
            low=m.group(2).lower(); hit=[o for o in OPT[i] if o in low]
            out[i]=hit[-1] if hit else 'other'
    return out
def load(dirs,ids,minitems=6):
    reps=collections.defaultdict(lambda: collections.defaultdict(list))
    for d in dirs:
        for l in open(d if d.endswith('.jsonl') else d+'/answers.jsonl'):
            try: r=json.loads(l)
            except ValueError: continue
            if not isinstance(r.get('response'),str) or r.get('id') not in ids: continue
            p=parse_t1(r['response']) if r['id']=='t1' else {i:norm(v) for i,v in A.parse(r['response']).items()}
            if len(p)>=minitems: reps[r['endpoint']][r['id']].append(p)
    return reps
def phi(replies):
    c=collections.defaultdict(collections.Counter); n=collections.Counter()
    for p in replies:
        for i,v in p.items(): c[i][v]+=1; n[i]+=1
    return {i:{v:k/n[i] for v,k in c[i].items()} for i in c}
def T(ra,rb):
    pa,pb=phi(ra),phi(rb); t=0
    for i in set(pa)&set(pb):
        ks=set(pa[i])|set(pb[i]); t+=sum((pa[i].get(k,0)-pb[i].get(k,0))**2 for k in ks)
    return t
def pair(reps,a,b,B=300,rng=random.Random(0)):
    obs=0; null=[0.0]*B
    for pid in set(reps[a])&set(reps[b]):
        ra,rb=reps[a][pid],reps[b][pid]; obs+=T(ra,rb); pool=ra+rb
        for k in range(B):
            rng.shuffle(pool); null[k]+=T(pool[:len(ra)],pool[len(ra):])
    m=sum(null)/B; sd=(sum((x-m)**2 for x in null)/B)**.5 or 1e-9
    return (obs-m)/sd, sum(x>=obs for x in null)/B, obs
def show(eps,reps,lab):
    print('replies:',{lab(e):sum(len(v) for v in reps[e].values()) for e in eps})
    Z={}
    for a,b in itertools.combinations(eps,2): Z[a,b]=Z[b,a]=pair(reps,a,b)
    print('z of distance vs permutation null (high=different); *=p<0.01')
    print('      '+' '.join(f'{lab(e):>6}' for e in eps))
    for a in eps: print(f'{lab(a):>5} '+' '.join('     .' if a==b else f'{Z[a,b][0]:5.1f}'+('*' if Z[a,b][1]<0.01 else ' ') for b in eps))
    print('raw distance T')
    for a in eps: print(f'{lab(a):>5} '+' '.join('     .' if a==b else f'{Z[a,b][2]:6.2f}' for b in eps))
    return Z
if __name__=='__main__':
    fam=sys.argv[1]; ids=sys.argv[2].split(','); dirs=sys.argv[3:]
    reps=load(dirs,ids)
    if fam=='refs': show([e for e in A.REFLAB if e in reps],reps,lambda e:A.REFLAB[e])
    else: show(['ep-'+x for x in A.FAM[fam].split()],reps,lambda e:e[-2:])
