"""Offline clustering of model endpoints from categorical answers.

Rows have the form (endpoint, prompt_id, category, answer). Only a None answer
is missing. Each endpoint pair is compared on prompts answered by both. Prompt
distances are averaged within a category, then shared categories are averaged
with equal weight. Diagnostics report exactly which categories contributed.

Family groups are formed by average linkage. The same procedure runs within
each family to produce model groups. Thresholds are inclusive distances in
[0, 1]. For E endpoints, R replies, P prompts and at most A distinct answers
per endpoint/prompt, time is O(R + E² P A + E³), apart from sorting endpoint,
prompt and category identifiers. Pair diagnostics use O(E² C) space for C
categories. The clustering algorithm uses O(E²) auxiliary space.
"""

import math
from collections import Counter, defaultdict
from itertools import combinations


def _probabilities(counts, method):
    if method == "mode":
        highest = max(counts.values())
        modes = [answer for answer, count in counts.items() if count == highest]
        return dict.fromkeys(modes, 1.0 / len(modes))
    total = sum(counts.values())
    return {answer: count / total for answer, count in counts.items()}


def _prompt_distance(first, second, method):
    # For tied modes use total variation between uniform modal distributions.
    if method == "mode":
        return 1.0 - len(first.keys() & second.keys()) / max(len(first), len(second))

    # The base-2 Jensen-Shannon distance is the square root of mean KL.
    # Only observed answers contribute, so a single observation is well-defined.
    def divergence(source, other):
        return math.fsum(
            mass * math.log2(2.0 * mass / (mass + other.get(answer, 0.0)))
            for answer, mass in source.items()
        )

    js = (divergence(first, second) + divergence(second, first)) / 2.0
    return math.sqrt(max(0.0, min(1.0, js)))


def _average_linkage(endpoints, distances, threshold):
    # Initial IDs follow lexical endpoint order. Merge IDs increase, providing a
    # fixed tie-break even when many pairs have exactly the same distance.
    members = {index: [endpoint] for index, endpoint in enumerate(endpoints)}
    active_distances = {
        (a, b): distances[endpoints[a], endpoints[b]]
        for a, b in combinations(members, 2)
    }
    next_id = len(members)
    while active_distances:
        first, second = min(
            active_distances, key=lambda pair: (active_distances[pair], pair)
        )
        if active_distances.pop((first, second)) > threshold:
            break
        left, right = members.pop(first), members.pop(second)
        size = len(left) + len(right)
        for other in members:
            a = active_distances.pop(tuple(sorted((first, other))))
            b = active_distances.pop(tuple(sorted((second, other))))
            active_distances[other, next_id] = (len(left) * a + len(right) * b) / size
        members[next_id] = left + right
        next_id += 1
    return sorted(sorted(group) for group in members.values())


def cluster(records, method, family_threshold, model_threshold):
    """Return sorted (endpoint, family, model) rows and JSON-safe diagnostics.

    'mode' compares all tied most-frequent answers. 'js' uses the empirical
    frequencies of all valid answers. Raises ValueError when any endpoint lacks
    usable answers, any pair has zero shared valid prompts, or the input
    parameters are inconsistent. A one-endpoint dataset forms one family/model.
    """
    if method not in ("mode", "js"):
        raise ValueError("method must be 'mode' or 'js'")
    for name, value in (("family_threshold", family_threshold),
                        ("model_threshold", model_threshold)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite distance from 0 to 1")

    counts = {}
    categories = {}
    for endpoint, prompt_id, category, answer in records:
        if not all(isinstance(field, str) and field for field in (endpoint, prompt_id, category)):
            raise ValueError("endpoint, prompt_id and category must be nonempty strings")
        if categories.setdefault(prompt_id, category) != category:
            raise ValueError(f"prompt {prompt_id!r} appears in multiple categories")
        prompt_counts = counts.setdefault(endpoint, {})
        if answer is None:
            continue
        if not isinstance(answer, str):
            raise ValueError("an answer must be a string or None")
        if prompt_id not in prompt_counts:
            prompt_counts[prompt_id] = Counter()
        prompt_counts[prompt_id][answer] += 1

    endpoints = sorted(counts)
    if not endpoints:
        raise ValueError("no endpoint records")
    empty = [endpoint for endpoint in endpoints if not counts[endpoint]]
    if empty:
        raise ValueError("no usable answers for endpoint(s): " + ", ".join(empty))
    profiles = {
        endpoint: {prompt: _probabilities(counter, method)
                   for prompt, counter in prompt_counts.items()}
        for endpoint, prompt_counts in counts.items()
    }
    distances = {}
    pairs = []
    for left, right in combinations(endpoints, 2):
        first, second = profiles[left], profiles[right]
        shared = first.keys() & second.keys()
        if not shared:
            raise ValueError(f"endpoints {left!r} and {right!r} have no shared valid prompts")
        by_category = defaultdict(list)
        for prompt in sorted(shared):
            by_category[categories[prompt]].append(
                _prompt_distance(first[prompt], second[prompt], method)
            )
        category_means = {name: math.fsum(values) / len(values)
                          for name, values in sorted(by_category.items())}
        distance = math.fsum(category_means.values()) / len(category_means)
        distances[left, right] = distance
        pairs.append({
            "left": left, "right": right, "distance": distance,
            "shared_prompts": len(shared),
            "category_prompts": {name: len(by_category[name]) for name in category_means},
            "category_distances": category_means,
        })

    rows = []
    family_sizes = {}
    model_sizes = {}
    families = _average_linkage(endpoints, distances, family_threshold)
    for family_number, family in enumerate(families, 1):
        family_label = f"F{family_number:02d}"
        family_sizes[family_label] = len(family)
        for model_number, model in enumerate(_average_linkage(family, distances, model_threshold), 1):
            model_label = f"{family_label}-M{model_number:02d}"
            model_sizes[model_label] = len(model)
            rows.extend((endpoint, family_label, model_label) for endpoint in model)
    return sorted(rows), {
        "method": method, "family_threshold": family_threshold, "model_threshold": model_threshold,
        "usable_prompts": {endpoint: len(profiles[endpoint]) for endpoint in endpoints},
        "pairs": pairs, "family_sizes": family_sizes, "model_sizes": model_sizes,
    }
