"""Group variants by the magnitude and within-question rank of response deviations.

Given a proposed family, compare each response with other endpoints on the
same questions. Votes use answer grammar, option copying and the order of
the answer and explanation. Columns with little evidence get smaller votes.
The upper-third cutoff is an exploratory prior shared with the earlier sequence
method. Returned labels are predictions; the program makes no network calls.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from sequence import OPT, observation, order

TOPIC_STOPWORDS = set('a the in of and to is as only reply with'.split())


@lru_cache(maxsize=8192)
def surface_events(prompt, text):
    """Summarize choice/explanation structure for one reply."""
    text = ' '.join(text.split())
    length = len(re.findall(r"[a-zA-Z']+|\d+", text.lower()))
    options = OPT.search(prompt)
    if options:
        lead = re.match(r'^\(?([A-D])\)?(?:\b|[):;.])', text)
        if not lead:
            return (0, 0, 0, 0, 1, min(2., np.log1p(length) / 2), 0, 0, 0, 0)
        after = text[lead.end():].lstrip(' .:);')
        literal = options.groups()[ord(lead[1]) - ord('A')].strip('.,;: ').lower()
        copied = after.casefold().startswith(literal)
        multiple_labels = len(set(re.findall(r'\b([A-D]):', after))) > 1
        return (after.endswith('.'), multiple_labels, length > 8, length > 3 and not copied, 0, 0, 0, 0, 0, 0)

    numeric_start = bool(re.match(r'^[-+]?\d+', text))
    enumeration = bool(re.search(r'(?:\(1\)|1\.[ ])[^\n]*(?:\(2\)|2\.[ ])', text))
    reasoning = bool(re.search(r'\b(?:therefore|explanation|because|for example|this means)\b', text, re.I))
    parts = re.split(r'[.!?]\s+(?=[A-Z])', text, maxsplit=1)
    tail_length = len(parts[1].split()) if len(parts) > 1 else 0
    prompt_words = set(re.findall('[a-z]+', prompt.lower())) - TOPIC_STOPWORDS
    words = re.findall(r"[a-zA-Z']+|\d+", text.lower())
    content = [word for word in words if word.isalpha() and word not in TOPIC_STOPWORDS]
    overlap = sum(word in prompt_words for word in content) / max(1, len(content))
    restated = not numeric_start and length >= 4 and not tail_length
    return (0, 0, 0, 0, 0, 0, enumeration, reasoning, min(2., np.log1p(tail_length) / 2), restated * overlap)


def grammar_contributions(features, formats):
    """Conditional rate of each grammar event versus peers on the same prompt."""
    successes = np.nansum(features, axis=2)
    opportunities = np.sum(~np.isnan(features), axis=2)
    peer_successes = successes.sum(axis=0, keepdims=True) - successes
    peer_opportunities = opportunities.sum(axis=0, keepdims=True) - opportunities
    expected = np.zeros_like(successes)
    for kind in set(formats):
        mask = formats == kind
        sy = peer_successes[:, mask]
        sn = peer_opportunities[:, mask]
        prior = (sy.sum(axis=1, keepdims=True) - sy + .5) / (sn.sum(axis=1, keepdims=True) - sn + 1.)
        expected[:, mask] = (sy + 4 * prior) / (sn + 4)
    residual = (successes - opportunities * expected).sum(axis=-1)
    uncertainty = np.sqrt(1 + (opportunities * expected * (1 - expected)).sum(axis=(1, 2)))
    return residual / uncertainty[:, None]


def predict(rows):
    """Predict one label per endpoint in one family of balanced cached replies."""
    endpoints = sorted({row['endpoint'] for row in rows})
    questions = sorted({row['id'] for row in rows})
    if not endpoints:
        raise ValueError('No replies supplied')
    cells = defaultdict(list)
    prompts = {}
    seen = set()
    for row in rows:
        endpoint, q, repeat = row['endpoint'], row['id'], row['repeat']
        slot = (endpoint, q, repeat)
        if slot in seen or prompts.setdefault(q, row['prompt']) != row['prompt']:
            raise ValueError(f'Duplicate reply or inconsistent prompt at {slot}')
        if row.get('synthetic') or not isinstance(row['response'], str) or not row['response'].strip():
            raise ValueError(f'Expected a real non-empty reply at {slot}')
        seen.add(slot)
        cells[endpoint, q].append(row['response'])
    repeats = len(cells[endpoints[0], questions[0]])
    shape = (len(endpoints), len(questions))
    features = np.full((*shape, repeats, 11), np.nan)
    ordering = np.empty((*shape, 2))
    # Center each correction coordinate independently, before combining them.
    surface = np.empty((*shape, 10))
    formats = np.array(['choice' if OPT.search(prompts[q]) else 'integer' for q in questions])
    for i, endpoint in enumerate(endpoints):
        for j, q in enumerate(questions):
            answers, prompt = cells[endpoint, q], prompts[q]
            if len(answers) != repeats or not repeats:
                raise ValueError('All endpoints must have the same number of replies for every selected prompt')
            features[i, j] = [observation(q, prompt, text) for text in answers]
            ordering[i, j] = np.mean([order(prompt, text) for text in answers], axis=0)
            surface[i, j] = np.mean([surface_events(prompt, text) for text in answers], axis=0)
    base = grammar_contributions(features, formats)
    ordering -= np.median(ordering, axis=0)
    order_effect = (ordering[:, :, 0] - ordering[:, :, 1]) / np.sqrt(len(questions) / 5)
    surface -= np.median(surface, axis=0)
    weights = np.array([1., 1., 1., 1., -1., -.5, 1., 1., 1., -1.])
    format_counts = np.array([sum(formats == kind) for kind in formats])
    correction = 2 * np.einsum('eqf,f->eq', surface, weights) / format_counts[None, :]
    per_prompt = base + order_effect + correction
    cap = np.quantile(np.abs(per_prompt), .90, axis=1, keepdims=True)
    robust = (np.clip(per_prompt, -cap, cap) + per_prompt - order_effect * (formats == 'integer')[None, :]) / 2
    ranks = rankdata(np.round(robust, 10), axis=0)
    ranks = (ranks - (len(endpoints) + 1) / 2) / max(1, (len(endpoints) - 1) / 2)
    spread = np.std(robust, axis=0)
    scores = (np.arcsinh(robust) + ranks * (spread / (spread + .25))).sum(axis=1)
    cutoff = np.quantile(scores, 2 / 3)
    return {ep: ('core' if s <= cutoff + 1e-12 else ep) for ep,s in zip(endpoints, scores)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--answers', required=True, type=Path)
    parser.add_argument('--families', type=Path, help='Formal endpoint,family CSV')
    parser.add_argument('--prompts', type=Path, default=Path(__file__).with_name('prompts.csv'))
    parser.add_argument('--split', choices=('dev', 'holdout'))
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.resolve() in {p.resolve() for p in (args.answers, args.families, args.prompts) if p}:
        parser.error('Choose an output path separate from the inputs')
    with args.prompts.open(newline='', encoding='utf-8-sig') as stream:
        bank = {r['id']: r for r in csv.DictReader(stream)}
    replies = []
    with args.answers.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            q = bank[row['id']]
            if q['prompt'] != row['prompt']:
                parser.error('The prompt in the archive differs from the prompt bank')
            if not args.split or q['split'] == args.split:
                replies.append(row)
    endpoints = {r['endpoint'] for r in replies}
    if not endpoints:
        parser.error('No matching replies')
    if args.families:
        with args.families.open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        families = {r['endpoint']: r['family'] for r in rows}
        if len(rows) != len(families) or set(families) != endpoints or not all(families.values()):
            parser.error('Each endpoint needs one non-empty family label')
    elif all(e.startswith('ref-') for e in endpoints):
        families = dict.fromkeys(endpoints, 'P')
    else:
        parser.error('Formal endpoints require a family map')
    out = []
    for family in sorted(set(families.values())):
        part = predict([r for r in replies if families[r['endpoint']] == family])
        out.extend((e, family, f'{family}/{g}') for e, g in part.items())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('endpoint', 'family', 'model'))
        writer.writerows(sorted(out))


if __name__ == '__main__':
    main()
