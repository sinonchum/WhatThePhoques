"""Cluster cached replies by departures from the family's usual answer format.

Within each supplied family, the upper quartile of format-deviation scores is
a candidate for distinct model variants. All other endpoints share a candidate
identity. This assumption is an exploratory baseline, especially when endpoints
produce very short answers. No labels, reference models or network calls are used.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from math import floor
from pathlib import Path
from statistics import mean, median


def quantile(values, probability):
    values = sorted(values)
    pos = (len(values) - 1) * probability
    i = floor(pos)
    return values[i] + (pos - i) * (values[min(i + 1, len(values) - 1)] - values[i])


def signature(text, kind):
    text = ' '.join(text.split())
    multi_number = len(re.findall(r'\d+', text)) > 1
    if kind == 'integer':
        return (text.endswith('.'), multi_number)
    return (bool(re.match(r'^[A-D]:', text)), bool(re.match(r'^[A-D][.)]', text)),
            text.endswith('.'), text.count(':') > 1, multi_number)


def group_profiles(data, endpoints):
    views = defaultdict(list)
    for kind in ('choice', 'integer'):
        if not any(data.get((ep, kind)) for ep in endpoints):
            continue
        if any(not data.get((ep, kind)) for ep in endpoints):
            raise ValueError(f'A {kind} reply is missing for an endpoint in this family.')
        profiles = [[mean(col) for col in zip(*data[ep, kind])] for ep in endpoints]
        centers = [median(col) for col in zip(*profiles)]
        distances = [[abs(v - m) for v, m in zip(row, centers)] for row in profiles]
        scales = [max(quantile(col, .75), .15) for col in zip(*distances)]
        for ep, row in zip(endpoints, distances):
            views[ep].append(median(v / scale for v, scale in zip(row, scales)))
    scores = {ep: mean(views[ep]) for ep in endpoints}
    threshold = quantile(scores.values(), .75)
    return {ep: ep if value > threshold else 'shared' for ep, value in scores.items()}


def classify(answers, prompts, family_path, split):
    with prompts.open(newline='', encoding='utf-8-sig') as f:
        bank = {r['id']: r for r in csv.DictReader(f)}
    data = defaultdict(list)
    seen = set()
    slots = defaultdict(set)
    with answers.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ep, pid = r['endpoint'], r['id']
            metadata = bank[pid]
            if split and metadata['split'] != split:
                continue
            key = (ep, pid, r['repeat'])
            if key in seen or r['prompt'] != metadata['prompt'] or r.get('synthetic'):
                raise ValueError(f'Inconsistent cached reply for {key}')
            seen.add(key)
            if isinstance(r['response'], str) and r['response'].strip():
                slots[ep].add((pid, r['repeat']))
                data[ep, metadata['format']].append(signature(r['response'], metadata['format']))
    ends = sorted({k[0] for k in seen})
    if not ends or not slots[ends[0]] or any(slots[ep] != slots[ends[0]] for ep in ends):
        raise ValueError('All endpoints must have replies on the same selected prompt/repeat slots.')
    if family_path:
        with family_path.open(newline='', encoding='utf-8-sig') as f:
            entries = list(csv.DictReader(f))
            family = {r['endpoint']: r['family'] for r in entries}
            if len(family) != len(entries):
                raise ValueError('A family map contains repeated endpoints.')
        if set(family) != set(ends) or not all(family.values()):
            raise ValueError('Family map must assign every endpoint exactly one family.')
    elif all(ep.startswith('ref-') for ep in ends):
        family = dict.fromkeys(ends, 'P')
    else:
        raise ValueError('Provide a family map for formal replies.')
    out = []
    for name in sorted(set(family.values())):
        members = [ep for ep in ends if family[ep] == name]
        labels = group_profiles(data, members)
        out.extend((ep, name, f'{name}/{labels[ep]}') for ep in members)
    return sorted(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--answers', type=Path, required=True)
    p.add_argument('--prompts', type=Path, default=Path(__file__).with_name('prompts.csv'))
    p.add_argument('--families', type=Path, help='CSV with endpoint,family columns')
    p.add_argument('--split', choices=['dev', 'holdout'])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.resolve() in {p.resolve() for p in (a.answers, a.prompts, a.families) if p}:
        p.error('Choose a separate output file.')
    rows = classify(a.answers, a.prompts, a.families, a.split)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['endpoint', 'family', 'model'])
        writer.writerows(rows)


if __name__ == '__main__':
    main()
