import numpy as np
import pandas as pd
from enum import Enum
from typing import List
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import CountVectorizer
from salt.constants import TEXT, VECTOR, CLUSTER, MEAN_DISTANCE


class DistanceType(Enum):
    LEXICAL = "Lexical"
    SEMANTIC = "Semantic"
    MIXED = "Mixed"


MAX_EXAMPLES = 10_000


def get_lexical_distances(texts: List[str]) -> np.array:
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(texts)
    return cosine_distances(vectors).astype(np.float16)


class Clusters:
    def __init__(self, df: pd.DataFrame):
        df_sample = df.sample(MAX_EXAMPLES, random_state=0) if len(df) > MAX_EXAMPLES else df
        lexical_distances = get_lexical_distances(df_sample[TEXT].to_list())
        semantic_distances = cosine_distances(df_sample[VECTOR].to_list())
        self.type2distances = {
            DistanceType.LEXICAL: lexical_distances,
            DistanceType.SEMANTIC: semantic_distances,
            DistanceType.MIXED: (lexical_distances + semantic_distances) / 2,
        }

        self.distance_type = DistanceType.LEXICAL
        self.df = df_sample[[TEXT]].copy().reset_index(drop=True)
        self.cluster_dfs = None
        self.update_clusters([0] * len(self.df))

    @property
    def num_examples(self):
        return len(self.df)

    @property
    def num_clusters(self):
        return len(self.cluster_dfs)

    @property
    def distances(self) -> np.array:
        return self.type2distances[self.distance_type]

    def get_mean_distances(self, df: pd.DataFrame):
        return [np.mean(row_distances[df.index]) for row_distances in self.distances[df.index]]

    def update_clusters(self, clusters: List[int]):
        self.df[CLUSTER] = clusters
        cluster2index = {cluster: i + 1 for i, cluster in enumerate(self.df[CLUSTER].value_counts().keys())}
        self.df[CLUSTER] = self.df[CLUSTER].apply(lambda cluster: cluster2index[cluster])

        for cluster, df_cluster in self.df.groupby(CLUSTER):
            mean_distances = self.get_mean_distances(df_cluster)
            self.df.loc[df_cluster.index, MEAN_DISTANCE] = mean_distances

        cluster2df = {cluster: df.sort_values(MEAN_DISTANCE) for cluster, df in self.df.groupby(CLUSTER)}
        self.cluster_dfs = [
            df for cluster, df in sorted(cluster2df.items(), key=lambda item: len(item[1]), reverse=True)
        ]

    def run(
        self,
        num_clusters=None,
        distance_threshold=None,
        distance_type=DistanceType.LEXICAL,
    ):
        if sum(x is None for x in [num_clusters, distance_threshold]) != 1:
            raise ValueError('You must specify either "num_clusters" or "distance_threshold"')

        self.distance_type = distance_type
        clusters = (
            AgglomerativeClustering(
                n_clusters=num_clusters,
                metric="precomputed",
                distance_threshold=distance_threshold,
                linkage="complete",
            )
            .fit(self.distances)
            .labels_
        )
        self.update_clusters(clusters)

    def get_data(self, cluster_index=None):
        if cluster_index is None:
            return pd.concat(self.cluster_dfs)
        return self.cluster_dfs[cluster_index - 1][[TEXT]]
