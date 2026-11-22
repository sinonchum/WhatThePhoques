"""Calibrate low-dimensional identity scores on actual pairwise practice MCC."""
import csv,hashlib,itertools,json,random
from pathlib import Path
import numpy as np
TASK = Path(__file__).resolve().parent
from metric_partition import edit_features
from replay import load_inputs
from variant_rank import predict as ranker

def value(comps,cfg):
    fields,prior=comps;w,b,c,cut=cfg
    score=(fields[:,:,:,0]+b*fields[:,:,:,1]+c*fields[:,:,:,2]).sum(axis=2)/(fields.shape[2]**.5)+w*prior
    mask=score>np.quantile(score,cut,axis=1,keepdims=True)+1e-12
    return mask

def collect(proto,draws,seed):
    bank,eps,cache=load_inputs()
    truth={r['endpoint']:r['model'] for r in csv.DictReader((TASK/'checkpoint/practice_answers.csv').open())}
    first=set(json.loads((TASK/'checkpoint/practice-replay.json').read_text())['first_ten'])
    ids={'dev':sorted(first),'holdout':sorted(q for q in bank if bank[q]['split']=='holdout'),'fixed-hard':sorted(q for q in bank if bank[q]['split']=='dev' and q not in first)}
    fields=[];prior=[]
    for j in range(draws):
        rng=random.Random(int.from_bytes(hashlib.blake2s(f'{seed}/{proto}/{j}'.encode(),digest_size=8).digest(),'big'))
        if proto=='stratified':
            qs=sorted(q for cat in sorted({r['category'] for r in bank.values()}) for q in rng.sample(sorted(q for q in bank if bank[q]['category']==cat),2))
        else:qs=ids[proto]
        rows=[cache[e,q,rng.randrange(3)] for e in eps for q in qs]
        fields.append([[edit_features(r['prompt'],r['response']) for r in rows[i:i+len(qs)]] for i in range(0,len(rows),len(qs))])
        aliases=['anonymous-%02d'%i for i in range(1,len(eps)+1)];rng.shuffle(aliases);ids_map=dict(zip(eps,aliases))
        blinded=[dict(r,endpoint=ids_map[r['endpoint']],repeat=0) for r in rows];rng.shuffle(blinded)
        v=ranker(blinded);prior.append([v[a]==a for a in aliases])
    return (np.array(fields),np.array(prior)), np.array([truth[e]==truth[f] for e,f in itertools.combinations(eps,2)])

def metric(comps,truth,cfg):
    mask=value(comps,cfg);l,r=np.triu_indices(mask.shape[1],1)
    pred=~(mask[:,l]|mask[:,r]);n=len(truth);p=truth.sum();k=pred.sum(axis=1)
    tp=(pred&truth).sum(axis=1);den=np.sqrt(p*(n-p)*k*(n-k))
    mcc=np.maximum(0,np.divide(tp*n-p*k,den,out=np.zeros(len(k)),where=den>0))
    return {'passes':int(sum(mcc>.65)),'mean':float(mcc.mean()),'median':float(np.median(mcc)),'fp':float((k-tp).mean()),'fn':float((p-tp).mean())}

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode",choices=("train","stratified","holdout","fixed-hard"))
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--configuration",type=Path)
    args=ap.parse_args()
    if args.mode!="train" and args.configuration is None:
        ap.error("Replay requires --configuration pointing to a training result")
    mode=args.mode;out=args.out;base=(.5,0,0,2/3)
    if mode=='train':
        datasets=[collect(k,512,420991) for k in ['dev','fixed-hard']]
        ref=[metric(*d,base) for d in datasets];candidates=[]
        for cfg in itertools.product([.25,.5,.75,1.],[0,.5,1.],[0,.5,1.,2.],[.60,2/3,.70]):
            stats=[metric(*d,cfg) for d in datasets]
            delta=[s['mean']-b['mean'] for s,b in zip(stats,ref)]
            candidates.append((min(delta),sum(delta),cfg,stats))
        candidates.sort(key=lambda v:(v[0],v[1]),reverse=True)
        print(candidates[0],flush=True)
        out.write_text(json.dumps({'count':len(candidates),'base':ref,'winner':candidates[0],'top5':candidates[:5]},indent=2)+'\n')
    else:
        cfg=json.loads(args.configuration.read_text())['winner'][2]
        d=collect(mode,2048,1493807)
        out.write_text(json.dumps({'protocol':mode,'seed':1493807,'draws':2048,'parameters':cfg,'old':metric(*d,base),'new':metric(*d,cfg)},indent=2)+'\n')
