import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from salt.logic.utils import (
    get_labels_from_str,
    get_str_from_labels,
    get_classes_from_labels,
)

THRESHOLD = 0.5


@dataclass
class Prediction:
    labels: List[str]
    class2probs: Dict[str, List[float]]


class Classifier(ABC):
    @abstractmethod
    def fit(self, vectors: List[List[float]], labels: List[str]) -> None:
        raise NotImplementedError()

    @abstractmethod
    def predict(self, vectors: List[List[float]]) -> Prediction:
        raise NotImplementedError()

    @abstractmethod
    def get_min_confidence_index(self, class2probs: Dict[str, List[float]]) -> int:
        raise NotImplementedError()


class SingleLabelClassifier(Classifier):
    def __init__(self):
        self.model = LogisticRegression(class_weight="balanced")

    def fit(self, vectors: List[List[float]], labels: List[str]) -> None:
        self.model.fit(vectors, labels)

    def predict(self, vectors: List[List[float]]) -> Prediction:
        classes = self.model.classes_
        vectors_probs = self.model.predict_proba(vectors)
        labels = classes[vectors_probs.argmax(axis=1)].tolist()
        class2probs = {cls: vectors_probs[:, index].tolist() for index, cls in enumerate(classes)}
        return Prediction(labels, class2probs)

    def get_min_confidence_index(self, class2probs: Dict[str, List[float]]) -> int:
        probs = np.stack(list(class2probs.values())).T
        return int(probs.max(axis=1).argmin())


class MultiLabelClassifier(Classifier):
    def __init__(self):
        self.model = MultiOutputClassifier(LogisticRegression(class_weight="balanced"))
        self.classes = None
        self.num_labels = None

    def fit(self, vectors: List[List[float]], labels: List[str]) -> None:
        self.num_labels = len(labels)
        self.classes = get_classes_from_labels(labels)

        class2index = {cls: index for index, cls in enumerate(self.classes)}
        labels_matrix = np.zeros((len(labels), len(class2index)))
        for i, labels_str in enumerate(labels):
            for label in get_labels_from_str(labels_str):
                labels_matrix[i, class2index[label]] = 1

        self.model.fit(vectors, labels_matrix)

    def predict(self, vectors: List[List[float]]) -> Prediction:
        vectors_probs = np.array([label_probs[:, 1] for label_probs in self.model.predict_proba(vectors)]).T

        labels = []
        for probs in vectors_probs:
            pred_classes = [self.classes[index] for index in np.argwhere(probs > THRESHOLD).flatten()]
            labels.append(get_str_from_labels(pred_classes) if pred_classes else self.classes[probs.argmax()])

        class2probs = {cls: vectors_probs[:, index].tolist() for index, cls in enumerate(self.classes)}
        return Prediction(labels, class2probs)

    def get_min_confidence_index(self, class2probs: Dict[str, List[float]]) -> int:
        focus_class = self.classes[self.num_labels % len(self.classes)]
        return int(abs(np.array(class2probs[focus_class]) - THRESHOLD).argmin())


def create_classifier(is_multilabel: bool) -> Classifier:
    return MultiLabelClassifier() if is_multilabel else SingleLabelClassifier()
