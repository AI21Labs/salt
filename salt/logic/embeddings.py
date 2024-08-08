import pickle
import numpy as np
import pandas as pd
import streamlit as st
from stqdm import stqdm
from pathlib import Path
from itertools import chain
from typing import Dict, List
from salt.constants import NA
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer


TEXTS_KEY = "texts"
VECTORS_KEY = "vectors"
LABELS_KEY = "labels"
METADATA_KEY = "metadata"
MODEL_NAME_KEY = "model_name"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


@st.cache_resource
def get_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def load_embeddings(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_embeddings(embeddings: Dict, path: str) -> None:
    Path(path).parent.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        return pickle.dump(embeddings, f)


def get_relevant_texts(texts: list[str]) -> list[str]:
    return list(set([text for text in texts if len(text.strip()) > 0]))


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model(MODEL_NAME)
    texts_sentences = [sent_tokenize(text) for text in texts]
    texts_lengths = [len(sentences) for sentences in texts_sentences]
    all_sentences = list(chain.from_iterable(texts_sentences))
    if len(all_sentences) <= BATCH_SIZE:
        sentences_vectors = model.encode(all_sentences).tolist()
    else:
        sentences_chunks = np.array_split(all_sentences, len(all_sentences) // BATCH_SIZE)
        sentences_vectors = list(
            chain.from_iterable(
                [
                    model.encode(sentences).tolist()
                    for sentences in stqdm(
                        sentences_chunks,
                        desc="vectorizing texts",
                        unit_scale=BATCH_SIZE,
                    )
                ]
            )
        )
    texts_vectors = []
    start_sentence_index = 0
    while start_sentence_index < len(sentences_vectors):
        end_sentence_index = start_sentence_index + texts_lengths[len(texts_vectors)]
        vector = np.mean(sentences_vectors[start_sentence_index:end_sentence_index], axis=0)
        texts_vectors.append(vector.tolist())
        start_sentence_index = end_sentence_index

    return texts_vectors


def create_embeddings(df: pd.DataFrame, text_column: str, label_column: str = None) -> Dict:
    data = df.fillna("").astype("str").to_dict("records")
    texts = get_relevant_texts([x[text_column] for x in data])
    vectors = embed_texts(texts)

    if label_column:
        text2label = {x[text_column]: x[label_column] if len(x[label_column].strip()) > 0 else NA for x in data}
        labels = [text2label[text] for text in texts]
    else:
        labels = [NA] * len(texts)

    return get_embeddings_dict(texts, vectors, labels, MODEL_NAME)


def get_embeddings_dict(
    texts: List[str],
    vectors: List[List[float]],
    labels: List[str],
    model_name: str = MODEL_NAME,
) -> Dict:
    return {
        TEXTS_KEY: texts,
        VECTORS_KEY: vectors,
        LABELS_KEY: labels,
        METADATA_KEY: {MODEL_NAME_KEY: model_name},
    }


def create_and_dump_embeddings(
    df: pd.DataFrame,
    text_column: str,
    label_column: str = None,
    output_path: str = None,
):
    output_path = output_path or "embeddings.pkl"
    embeddings = create_embeddings(df, text_column, label_column)
    dump_embeddings(embeddings, output_path)
