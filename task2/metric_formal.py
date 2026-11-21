"""Pool within-family pairs to optimize a surrogate for the official model MCC."""
import argparse
import csv
import json
from itertools import combinations, product
from pathlib import Path

import numpy as np

from choice_edit import CHECKPOINT, predict as syntax_baseline
from variant_rank import predict as submitted_rule
from short_answers import predict as content_baseline
from metric_partition import candidates, clipped_scores, content_scenarios, features, regularized_scenarios


def family_data(rows, label, rng, view_rng):
    eps = sorted({r["endpoint"] for r in rows})
    components = features(rows)
    syntax = regularized_scenarios(components)
    content = content_scenarios(rows)
    atomic = sum(len(r["response"].split()) == 1 for r in rows) / len(rows)
    draws = 192
    # Both views are hypotheses. Give syntax more weight in longer-answer families.
    n_content = draws//2 if atomic >= .8 else draws//4
    hypotheses = np.concatenate((syntax[rng.integers(len(syntax), size=draws-n_content)],
                                 content[rng.integers(len(content), size=n_content)]))
    rng.shuffle(hypotheses)
    groups, masks, truth = candidates(hypotheses)
    old = (content_baseline if label == "G2" else syntax_baseline)(rows)
    codes = np.unique([old[e] for e in eps], return_inverse=True)[1]
    left,right = np.triu_indices(len(eps),1)
    old_mask = codes[left]==codes[right]
    posted = submitted_rule(rows)
    posted_labels = np.array([posted[e] for e in eps])
    posted_mask = posted_labels[left]==posted_labels[right]
    if not np.any(np.all(masks==old_mask,axis=1)):
        groups = np.vstack((groups,codes))
        masks = np.vstack((masks,old_mask))
    # Keep uncertainty about the evidence model separate from prompt uncertainty.
    views = [truth]
    for pool in (syntax, content):
        sampled = pool[view_rng.integers(len(pool),size=draws)]
        views.append(sampled[:,left]==sampled[:,right])
    truth = np.concatenate(views,axis=0)
    return {"endpoints":eps, "groups":groups, "masks":masks, "truth":truth,
            "old":old_mask, "posted":posted_mask, "cross":masks.astype(int) @ truth.T.astype(int)}


def optimize(blocks):
    # For N eligible pairs, P true positives-in-ground-truth, K predicted same:
    # MCC = (N*TP - P*K) / sqrt(P*(N-P)*K*(N-K)).
    truth = np.concatenate([b["truth"] for b in blocks],axis=1)
    p = truth.sum(axis=1)
    n = truth.shape[1]
    root_p = np.sqrt(p*(n-p))
    utilities = []
    calls = 0
    sizes = [b["masks"].sum(axis=1) for b in blocks]
    # Vectorize across the last family's candidate cuts.
    for prefix in product(*(range(len(b["groups"])) for b in blocks[:-1])):
        tp = blocks[-1]["cross"].copy()
        k = sizes[-1].copy()
        for i,j in enumerate(prefix):
            tp += blocks[i]["cross"][j]
            k += sizes[i][j]
        denom = np.sqrt(k*(n-k))[:,None] * root_p
        corr = np.divide(n*tp-k[:,None]*p, denom, out=np.zeros_like(denom), where=denom>0)
        utilities.append(np.maximum(corr,0.).reshape(len(k),3,-1).mean(axis=2))
        calls += len(k)
    # Minimax regret compares to the best reachable partition under each view,
    # so high-agreement but uninformative views do not set the scale alone.
    utility = np.vstack(utilities)
    regret = (utility.max(axis=0) - utility).max(axis=1)
    shape = tuple(len(b["groups"]) for b in blocks)
    picks = {"mixed":int(utility[:,0].argmax()),"minimax":int(regret.argmin())}
    return {k:np.unravel_index(v,shape) for k,v in picks.items()}, calls



def run(records, families, out):
    panels = json.loads((CHECKPOINT/"practice-replay.json").read_text())
    which = {"round1":set(panels["first_ten"]), "round2":set(panels["additional_ten"])}
    which["combined"] = which["round1"] | which["round2"]
    fits = {}
    report = {}
    for i,(name,qs) in enumerate(which.items()):
        rng = np.random.default_rng(605407+i)
        labels = sorted(set(families.values()))
        blocks = [family_data([r for r in records if r["id"] in qs and families[r["endpoint"]]==f], f, rng, np.random.default_rng(19487+i*31+j)) for j,f in enumerate(labels)]
        selections,calls = optimize(blocks)
        old = np.concatenate([b["old"] for b in blocks])
        truth = np.concatenate([b["truth"] for b in blocks],axis=1)
        predictions = {"previous":old, "submitted_rule":np.concatenate([b["posted"] for b in blocks])}
        for policy,ix in selections.items():
            predictions[policy] = np.concatenate([b["masks"][j] for b,j in zip(blocks,ix)])
            if name=="combined":
                with (out/(policy+".csv")).open("w",newline="") as stream:
                    writer=csv.writer(stream)
                    writer.writerow(("endpoint","family","model"))
                    result=[(e,f,f"{f}/{g}") for f,b,j in zip(labels,blocks,ix)
                            for e,g in zip(b["endpoints"],b["groups"][j])]
                    writer.writerows(sorted(result))
        fits[name]=(predictions,truth)
        report[name]={"n_partitions_scored":calls, "eligible_pairs_assuming_families":int(len(old)),
                      "changed_pairs_vs_previous":{m:int(sum(old!=v)) for m,v in predictions.items()}}
    for name,other in [("round1","round2"),("round2","round1"),("combined","combined")]:
        preds,_=fits[name]
        evidence=fits[other][1]
        report[name]["scored_against_panel"]=other
        values=clipped_scores(list(preds.values()),evidence).reshape(len(preds),3,-1).mean(axis=2)
        report[name]["surrogate_mcc_by_view"]={k:dict(zip(("mixed","syntax","content"),v)) for k,v in zip(preds,values)}
    def pair_mask(path):
        with path.open(newline="") as stream:
            labels = {r["endpoint"]:(r["family"],r["model"]) for r in csv.DictReader(stream)}
        if set(labels)!=set(families):
            raise ValueError(f"Endpoint coverage differs in {path}")
        return np.array([labels[e]==labels[f] for e,f in combinations(sorted(labels),2)])
    snapshots = {"round2_submitted":CHECKPOINT/"submission.csv",
                 "previous_offline":CHECKPOINT.parent/"experiments/choice-edit/formal/candidate.csv"}
    report["combined"]["changed_pairs_all_528"] = {
        policy:{name:int(np.count_nonzero(pair_mask(out/(policy+".csv"))!=pair_mask(path)))
                for name,path in snapshots.items()} for policy in ("mixed","minimax")}
    (out/"formal.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    rows=[json.loads(l) for l in (CHECKPOINT/"formal.jsonl").read_text().splitlines()]
    with (CHECKPOINT/"families.csv").open(newline="") as stream:
        families={r["endpoint"]:r["family"] for r in csv.DictReader(stream)}
    args.out.mkdir(parents=True,exist_ok=False)
    run(rows,families,args.out)
