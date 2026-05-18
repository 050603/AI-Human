from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Iterable


def shannon_entropy(values: Iterable[Hashable]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p + 1e-12)
    return entropy
