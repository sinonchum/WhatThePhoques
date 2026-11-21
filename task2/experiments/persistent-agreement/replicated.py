"""Persistence-gated variant of the last submitted practice rule.

`metric_partition.predict` detaches the top third of endpoints by answer-format
score on the whole panel.  Here an endpoint is detached only when that rule
detaches it on BOTH of two disjoint question blocks (alternating questions in
id order, so a stratified ten-question panel gives one question per category in
each block).  No parameter is fitted; `union` is the mirror-image control.
"""
from common import TASK  # noqa: F401  (puts task2/ on sys.path)
from metric_partition import predict as last_rule


def blocks(rows):
    qs = sorted({r["id"] for r in rows})
    first = set(qs[0::2])
    return [r for r in rows if r["id"] in first], [r for r in rows if r["id"] not in first]


def _detached(rows):
    return {e for e, g in last_rule(rows).items() if g != "core"}


def predict(rows):
    a, b = (_detached(part) for part in blocks(rows))
    both = a & b
    return {e: e if e in both else "core" for e in {r["endpoint"] for r in rows}}


def predict_union(rows):
    a, b = (_detached(part) for part in blocks(rows))
    return {e: e if e in a | b else "core" for e in {r["endpoint"] for r in rows}}
