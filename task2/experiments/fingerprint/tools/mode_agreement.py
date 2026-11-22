import json,glob,collections,sys
sys.path.insert(0,'.')
from analyze_num import parse,FAM
D=collections.defaultdict(lambda: collections.defaultdict(list)); keys=set()
for f in glob.glob('data/claude-code-run/*.jsonl'):
    for l in open(f):
        try: r=json.loads(l)
        except: continue
        if r.get('id') not in ('n1','n2','n3','n4'): continue
        keys|=set(r.keys())
        txt=r.get('response') or ''
        if not isinstance(txt,str): txt=json.dumps(txt)
        for i,v in parse(txt).items(): D[r['endpoint']][(r['id'],i)].append(v)
print(sorted(keys))
def mode(lst):
    c=collections.Counter(lst).most_common(2)
    return c[0][0], c[0][1]/len(lst)
for g,m in FAM.items():
    es=['ep-'+x for x in m.split()]
    print(g,'samples/item',{e[-2:]:round(sum(len(v) for v in D[e].values())/max(1,len(D[e])),1) for e in es})
    print('     sharp '+' '.join(f"{sum(mode(v)[1] for v in D[e].values() if len(v)>=3)/max(1,sum(1 for v in D[e].values() if len(v)>=3)):.2f}" for e in es))
    print('      '+'    '.join(e[-2:] for e in es))
    for a in es:
        row=[]
        for b in es:
            its=[k for k in D[a] if len(D[a][k])>=3 and len(D[b].get(k,[]))>=3]
            # prob that b's distribution gives to a's mode
            s=sum(collections.Counter(D[b][k])[mode(D[a][k])[0]]/len(D[b][k]) for k in its)/max(1,len(its))
            row.append(f"{s:.2f}")
        print('  '+a[-2:]+'  '+'  '.join(row))
