"""Route mostly atomic families to a content-based identity heuristic."""
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import rankdata

from inputs import _normalize
from sequence import OPT
from choice_edit import predict as rank_predict


def predict(rows):
    # Freeze routing at the family level before comparing the question panels.
    if sum(len(r['response'].split()) == 1 for r in rows) < .8 * len(rows):
        return rank_predict(rows)
    endpoints = sorted({r['endpoint'] for r in rows})
    questions = sorted({r['id'] for r in rows})
    cells = defaultdict(list)
    for r in rows:
        choice = bool(OPT.search(r['prompt']))
        value = _normalize(r['response'], 'choice' if choice else 'integer', r['prompt'])
        if value is None and len(r['response'].split()) == 1:
            value = r['response'].strip().casefold()
        cells[r['endpoint'],r['id']].append(value)
    columns = []
    for q in questions:
        values = []
        for ep in endpoints:
            counts = Counter(v for v in cells[ep,q] if v is not None)
            peak = max(counts.values(), default=0)
            modes = [v for v in counts if counts[v] == peak]
            values.append(modes[0] if len(modes) == 1 else None)
        counts = Counter(v for v in values if v is not None)
        if len(counts) < 2:
            continue
        # Missing evidence receives a neutral vote. Rank rarity within prompt.
        present = np.array([i for i,v in enumerate(values) if v is not None])
        rarity = rankdata([-counts[values[i]] for i in present])
        col = np.zeros(len(endpoints))
        col[present] = (rarity-(1+len(present))/2) / len(present)
        columns.append(col)
    if not columns:
        return dict.fromkeys(endpoints,'core')
    scores = np.mean(columns,axis=0)
    threshold = np.quantile(scores,2/3)
    return {ep: ep if s>threshold+1e-12 else 'core' for ep,s in zip(endpoints,scores)}
