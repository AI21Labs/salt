import pandas as pd
from dataclasses import dataclass
from salt.logic.clusters import Clusters
from salt.logic.embeddings import embed_texts
from sklearn.metrics.pairwise import cosine_similarity
from salt.logic.utils import get_prob_col, get_labels_from_str
from salt.constants import TEXT, VECTOR, LABEL, PRED, CLUSTER, DATE


@dataclass
class FilterParams:
    label: str = None
    pred: str = None
    cluster: int = None
    query: str = None
    use_regex: bool = False
    use_semantics: bool = False


class Filter:
    def __init__(self, clusters: Clusters):
        self.clusters = clusters

    @staticmethod
    def sort_by_similarity(df: pd.DataFrame, query: str) -> pd.DataFrame:
        vector = embed_texts([query])
        df["similarity"] = cosine_similarity(vector, df[VECTOR].to_list()).flatten()
        return df.sort_values(by="similarity", ascending=False).drop(columns={"similarity"})

    def get_data(self, df: pd.DataFrame, params: FilterParams) -> pd.DataFrame:
        if params.label:
            df = df[df[LABEL].apply(lambda s: params.label in get_labels_from_str(s))]
        if params.cluster is not None:
            df_cluster = self.clusters.get_data(params.cluster)
            df = df[df[TEXT].isin(df_cluster[TEXT])]
        if params.pred:
            df = df[df[PRED].apply(lambda s: params.pred in get_labels_from_str(s))].sort_values(
                get_prob_col(params.pred), ascending=False
            )
        if params.query:
            if params.use_semantics:
                df = self.sort_by_similarity(df, params.query)
            else:
                df = df[df[TEXT].str.contains(params.query, case=False, regex=params.use_regex)]
        return df[[TEXT, LABEL, PRED, CLUSTER, DATE] + ([get_prob_col(params.pred)] if params.pred else [])]
