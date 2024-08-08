import os
import pandas as pd
from glob import glob
from datetime import datetime
from typing import Dict, Optional
from salt.logic.clusters import Clusters
from salt.logic.filter import Filter, FilterParams
from salt.logic.active_learning import ActiveLearningMechanism
from salt.constants import (
    TEXT,
    VECTOR,
    DATE,
    LABEL,
    PRED,
    PROB,
    CLUSTER,
    NA,
    PROJECTS_DIR,
    EMBEDDINGS_FILE_NAME,
)
from salt.logic.embeddings import (
    create_embeddings,
    dump_embeddings,
    load_embeddings,
    get_embeddings_dict,
    TEXTS_KEY,
    VECTORS_KEY,
    LABELS_KEY,
    METADATA_KEY,
    MODEL_NAME_KEY,
)


def get_working_dir(project_name: str) -> str:
    return os.path.join(PROJECTS_DIR, project_name)


def get_embeddings_filepath(project_name: str) -> str:
    return os.path.join(get_working_dir(project_name), EMBEDDINGS_FILE_NAME)


class SaltProject:
    def __init__(self, name: str, embeddings: Dict, df: pd.DataFrame = None):
        self.name = name
        self.df = SaltProject.init_state(embeddings, df)
        self.al = ActiveLearningMechanism(self.df)
        self.clusters = Clusters(self.df)
        self.filter = Filter(self.clusters)

    @property
    def num_annotations(self) -> int:
        return len(self.df[self.df[LABEL] != NA])

    @property
    def working_dir(self) -> str:
        return get_working_dir(self.name)

    @property
    def state_filename(self) -> str:
        return f"state_{self.num_annotations:06}.csv"

    @staticmethod
    def init_state(embeddings: Dict, df: pd.DataFrame = None):
        if df is None:
            df = pd.DataFrame({TEXT: embeddings[TEXTS_KEY], VECTOR: embeddings[VECTORS_KEY]})
            df[LABEL] = embeddings.get(LABELS_KEY, NA)  # backward-compatibility
            df[DATE] = datetime(1, 1, 1)
            df[PRED] = NA
            df[PROB] = NA
        else:
            df[VECTOR] = embeddings[VECTORS_KEY]
        df[CLUSTER] = NA
        return df

    def dump_state(self) -> None:
        self.df.drop(columns={VECTOR}).to_csv(f"{self.working_dir}/{self.state_filename}", index=False)

    @staticmethod
    def create(
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        project_name: str,
        base_project_name: Optional[str] = None,
    ) -> "SaltProject":

        if base_project_name:
            base_project = SaltProject.load(base_project_name)
            return base_project.extend(df, text_column, label_column, project_name)

        embeddings = create_embeddings(df, text_column, label_column)
        dump_embeddings(embeddings, get_embeddings_filepath(project_name))
        return SaltProject(project_name, embeddings)

    @staticmethod
    def load(project_name: str) -> "SaltProject":
        embeddings = load_embeddings(get_embeddings_filepath(project_name))
        state_files = sorted(glob(f"{get_working_dir(project_name)}/*.csv"))
        df = pd.read_csv(state_files[-1], na_filter=False) if state_files else None
        return SaltProject(project_name, embeddings, df)

    def extend(self, df: pd.DataFrame, text_column: str, label_column: str, project_name: str) -> "SaltProject":
        new_project = SaltProject.create(df, text_column, label_column, project_name)
        new_rows = [row for _, row in new_project.df.iterrows() if row[TEXT] not in self.df[TEXT]]

        df_extended = pd.concat([self.df, pd.DataFrame(new_rows)]).reset_index(drop=True)
        extended_embeddings = get_embeddings_dict(df_extended[TEXT], df_extended[VECTOR], df_extended[LABEL])
        dump_embeddings(extended_embeddings, get_embeddings_filepath(project_name))

        extended_project = SaltProject(project_name, extended_embeddings, df_extended)
        extended_project.dump_state()
        return extended_project

    def update_clusters(self, df_clusters: pd.DataFrame) -> None:
        text2cluster = {row[TEXT]: row[CLUSTER] for _, row in df_clusters.iterrows()}
        self.df[CLUSTER] = self.df[TEXT].apply(lambda text: text2cluster.get(text, NA))

    def get_data(self, params: FilterParams = None) -> pd.DataFrame:
        if not params:
            params = FilterParams()
        return self.filter.get_data(self.df, params)
