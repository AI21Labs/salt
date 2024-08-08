from typing import List
from functools import cache
from itertools import chain
from salt.constants import PROB, LABELS_SEP


@cache
def get_prob_col(label: str) -> str:
    return f"{PROB}_{label}"


@cache
def get_labels_from_str(labels_str: str) -> List[str]:
    return labels_str.split(LABELS_SEP)


def get_str_from_labels(labels: List[str]) -> str:
    return LABELS_SEP.join(sorted(labels))


def get_classes_from_labels(labels_strs: List[str]) -> List[str]:
    return sorted(set(chain.from_iterable([get_labels_from_str(s) for s in labels_strs])))
