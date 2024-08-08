import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from salt.logic.classifier import create_classifier, Prediction, Classifier
from salt.constants import NA, SKIP, LABEL, PRED, VECTOR, TEXT, DATE, LABELS_SEP
from salt.logic.utils import get_prob_col, get_labels_from_str, get_classes_from_labels


class ActiveLearningMechanism:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.model: Optional[Classifier] = None
        self.curr_ann_index: Optional[int] = None
        self.last_preds = [self.df[PRED]]
        self.history = []

    @property
    def labels(self) -> List[str]:
        labels = get_classes_from_labels(self.df[LABEL].dropna())
        return sorted([label for label in labels if label not in [NA, SKIP]])

    @property
    def label2index(self) -> Dict[str, int]:
        return {label: index for index, label in enumerate(self.labels)}

    @property
    def num_anns(self) -> int:
        return len(self.df[self.df[LABEL] != NA])

    @property
    def next_example(self) -> str:
        if self.curr_ann_index is None:
            self.step()
        return self.df.loc[self.curr_ann_index][TEXT]

    @property
    def all_labeled(self) -> bool:
        return self.num_anns == len(self.df)

    @property
    def is_multilabel(self) -> bool:
        return any(self.df[LABEL].str.contains(LABELS_SEP))

    @property
    def is_single_label_fittable(self) -> bool:
        return len(self.labels) > 1

    @staticmethod
    def is_multi_label_class_fittable(label, annotations) -> bool:
        return not all(label in get_labels_from_str(labels_str) for labels_str in annotations)

    def get_ann_options(self) -> List[str]:
        return self.labels + [SKIP]

    def get_train_df(self) -> pd.DataFrame:
        return self.df[~self.df[LABEL].isin([SKIP, NA])]

    def set_label(self, index: int, label: str) -> None:
        if self.df.loc[index, LABEL] == label:
            return

        self.df.loc[index, LABEL] = label
        self.df.loc[index, DATE] = datetime.now()
        self.curr_ann_index = None

    def set_labels(self, df: pd.DataFrame) -> None:
        df_original = self.df.loc[df.index]
        for _, row in df[df[LABEL] != df_original[LABEL]].iterrows():
            self.set_label(row.name, row[LABEL])

    def fit(self) -> None:
        df_train = self.get_train_df()
        self.model = create_classifier(self.is_multilabel)
        self.model.fit(df_train[VECTOR].to_list(), df_train[LABEL].to_list())

    def predict(self, df: pd.DataFrame) -> Prediction:
        vectors = df[VECTOR].to_list()
        return self.model.predict(vectors)

    def predict_and_update(self, df: pd.DataFrame) -> None:
        prediction = self.predict(df)
        df[PRED] = prediction.labels
        for index, cls in enumerate(self.labels):
            df[get_prob_col(cls)] = prediction.class2probs[cls]

    def step(self, label: str = None) -> None:
        if label:
            self.set_label(self.curr_ann_index, label)

        self.fit()
        self.predict_and_update(self.df)

        df_na = self.df[self.df[LABEL] == NA]
        if df_na.empty:
            self.curr_ann_index = None
            return

        class2probs = {cls: df_na[get_prob_col(cls)].to_list() for cls in self.labels}
        self.curr_ann_index = df_na.iloc[self.model.get_min_confidence_index(class2probs)].name

    def update_history_and_get_change_df(self) -> Optional[pd.DataFrame]:
        df_comp = pd.DataFrame(
            {
                "last": self.last_preds[0],
                "current": self.df[PRED].apply(get_labels_from_str),
            }
        )
        records = []
        for label in self.labels:
            df_label = df_comp[df_comp.apply(lambda row: label in row["last"] or label in row["current"], axis=1)]
            change_rate = (
                df_label["last"].apply(lambda labels: label in labels)
                != df_label["current"].apply(lambda labels: label in labels)
            ).mean()
            records.append(
                {
                    "num_labels": self.num_anns,
                    "class": label,
                    "change_rate": change_rate,
                }
            )
        self.last_preds.append(self.df[PRED].apply(get_labels_from_str).copy())
        if len(self.last_preds) > 10:
            self.last_preds.pop(0)
            self.history += records
            return pd.DataFrame(self.history)

        return None
