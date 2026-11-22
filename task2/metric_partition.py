"""Choose a transitive endpoint partition by its mean clipped pairwise MCC
against prompt-resampled working hypotheses. These hypotheses are unlabeled.
"""
import re
from collections import defaultdict
from functools import lru_cache
import numpy as np
from scipy.cluster.hierarchy import linkage, cut_tree
from choice_edit import evidence
from sequence import scores as sequence_scores
from variant_rank import predict as frozen_rank
from inputs import _normalize
from sequence import OPT


PRIOR_WEIGHT = 1.0
JITTER = 0.35


def clipped_scores(proposals, scenarios):
    """Rows of predictions × rows of pseudo-truth, columns = eligible pairs."""
    x = np.asarray(proposals, dtype=float)
    y = np.asarray(scenarios, dtype=float)
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(x, axis=1)[:, None] * np.linalg.norm(y, axis=1)
    corr = np.divide(np.einsum("ip,jp->ij",x,y), denom, out=np.zeros_like(denom), where=denom > 0)
    return np.maximum(0., corr)


@lru_cache(maxsize=64)
def weights(n, draws):
    # Same question multiplicities across endpoints. No endpoint IDs in features.
    rng = np.random.default_rng(11437 + n)
    return rng.multinomial(n, np.full(n, 1/n), size=draws).T


@lru_cache(maxsize=8192)
def transition(prompt, answer):
    text = answer.strip()
    if OPT.search(prompt):
        nonstandard = bool(re.match(r"^[(]?[A-D]\s*[,;=\]]",text))
        labels = set(re.findall(r"\b([A-D])\s*[:;.)]",text))
        return float(nonstandard) + float(len(labels)>1)
    # A concise assertion immediately followed by another sentence.
    return float(bool(re.match(r"^[^.!?\n]{1,80}[.!?]\s+[A-Z]",text)))


def features(rows, transition_weight=0.):
    eps, strength = sequence_scores(rows)
    prior = frozen_rank(rows)
    qs = sorted({r['id'] for r in rows})
    cells = defaultdict(list)
    for r in rows:
        cells[r['endpoint'],r['id']].append(evidence(r['prompt'], r['response']) + transition_weight*transition(r['prompt'],r['response']))
    grammar = np.array([[np.mean(cells[e,q]) for q in qs] for e in eps])
    prior = np.array([prior[e] == e for e in eps], dtype=float)
    strength = np.arcsinh(strength - np.median(strength))
    return eps, grammar, prior, strength


def core_groups(detached):
    # Positive IDs encode literal singletons; zero is the shared group.
    return detached.astype(int) * np.arange(1, detached.shape[1]+1)


def regularized_scenarios(components, prior_weight=PRIOR_WEIGHT, jitter=JITTER, draws=48):
    _, grammar, prior, strength = components
    n = grammar.shape[1]
    centered = grammar - grammar.mean(axis=0)
    base = centered.sum(axis=1) / np.sqrt(n)
    resampled = np.einsum("eq,qd->de",centered,weights(n,draws)) / np.sqrt(n)
    scores = base + jitter*(resampled-base) + prior_weight*prior
    # A weak continuous feature only resolves exact grammar/prior ties.
    scores = scores + .015*strength
    # Adjacent order statistics represent alternative core sizes.
    cuts = np.array([.57, 2/3, 2/3, 2/3, .73])
    thresholds = np.quantile(scores,cuts,axis=1).T.reshape(-1,1)
    return core_groups(np.repeat(scores,len(cuts),axis=0) > thresholds + 1e-12)


def content_scenarios(rows, draws=48):
    eps = sorted({r['endpoint'] for r in rows})
    qs = sorted({r['id'] for r in rows})
    cells = defaultdict(list)
    for r in rows:
        fmt = 'choice' if OPT.search(r['prompt']) else 'integer'
        cells[r['endpoint'],r['id']].append(_normalize(r['response'],fmt,r['prompt']))
    pair = np.triu_indices(len(eps),1)
    difference, observed = [], []
    for q in qs:
        vals = [cells[ep,q] for ep in eps]
        # Missing parses supply no equality evidence. Ignore unanimous prompts.
        distinct = {x for group in vals for x in group if x is not None}
        deltas, valid = [], []
        for i,j in zip(*pair):
            matches = [u!=v for u in vals[i] for v in vals[j] if u is not None and v is not None]
            has_info = bool(matches) and len(distinct)>1
            deltas.append(np.mean(matches) if has_info else 0.)
            valid.append(has_info)
        difference.append(deltas)
        observed.append(valid)
    w = weights(len(qs), draws).T
    # One pseudo-observation at distance 0.5 stabilizes sparse comparisons.
    dist = (.5 + np.einsum("dq,qp->dp",w,difference)) / (1 + np.einsum("dq,qp->dp",w,observed))
    samples = []
    for distances in dist:
        tree = linkage(distances, method='average')
        for k in (2,3,4):
            if k < len(eps):
                samples.append(cut_tree(tree,n_clusters=k).ravel())
    return np.asarray(samples)


def candidates(scenarios):
    """Search observed partitions plus every cut of the co-membership tree."""
    n = scenarios.shape[1]
    left,right = np.triu_indices(n,1)
    pairs = scenarios[:,left] == scenarios[:,right]
    tree = linkage(1.-pairs.mean(axis=0),method='average')
    cuts = cut_tree(tree).T
    groups = np.vstack([scenarios,cuts])
    masks = groups[:,left] == groups[:,right]
    _, unique = np.unique(masks,axis=0,return_index=True)
    return groups[unique],masks[unique],pairs


def solve(scenarios):
    groups,masks,pairs = candidates(scenarios)
    return groups[np.argmax(clipped_scores(masks,pairs).mean(axis=1))]


@lru_cache(maxsize=8192)
def edit_features(prompt, text):
    text = text.strip()
    trailing = bool(OPT.search(prompt) and re.match(r"^\(?[A-D][:.) ;\-]", text)
                    and text.endswith((".", ";")))
    return evidence(prompt,text), transition(prompt,text), int(trailing)


def predict(rows):
    """Practice-calibrated rule selected on the minimum mean-MCC gain in two panels."""
    prior = frozen_rank(rows)
    eps = sorted(prior)
    questions = sorted({r["id"] for r in rows})
    cells = defaultdict(list)
    for r in rows:
        base, shift, trailing = edit_features(r["prompt"],r["response"])
        cells[r["endpoint"],r["id"]].append(base + .5*shift + .5*trailing)
    scores = np.array([sum(np.mean(cells[e,q]) for q in questions) / np.sqrt(len(questions))
                       + .5*(prior[e]==e) for e in eps])
    cutoff = np.quantile(scores,2/3)
    return {e:e if v>cutoff+1e-12 else "core" for e,v in zip(eps,scores)}
