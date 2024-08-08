import pandas as pd
import streamlit as st
from typing import Optional
from itertools import chain
from collections import Counter
from salt.logic.project import SaltProject
from salt.constants import PROJECT_STATE_KEY, LABEL, PRED, NA, ALL
from salt.logic.utils import get_labels_from_str


def get_project_state() -> Optional[SaltProject]:
    if PROJECT_STATE_KEY not in st.session_state:
        st.error('Please go back to the "Setup" step to create or load a project.')
        return None

    return st.session_state[PROJECT_STATE_KEY]


def get_project_state_if_has_classes() -> Optional[SaltProject]:
    project = get_project_state()
    if not project:
        return
    al = project.al
    if al.is_multilabel:
        annotations = al.get_train_df()[LABEL]
        for label in project.al.labels:
            if not al.is_multi_label_class_fittable(label, annotations):
                st.error(f'Please add at least 1 negative example for the class "{label}".')
                return
    elif not al.is_single_label_fittable:
        st.error('Please define at least 2 classes in the "Review" step before proceeding into this step.')
        return
    return project


def get_counts_df(df: pd.DataFrame) -> pd.DataFrame:
    labels = get_project_state().al.get_ann_options() + [NA]
    col2values = {}
    for col in [LABEL, PRED]:
        counter = Counter(chain.from_iterable(df[col].apply(get_labels_from_str)))
        col2values[col] = [counter[label] for label in labels] + [len(df)]
    return pd.DataFrame(col2values, index=labels + [ALL])
