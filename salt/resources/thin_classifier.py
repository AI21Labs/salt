# Requirements:
# scikit-learn==1.3.2
# sentence-transformers==2.2.2

import pickle
import numpy as np
from tqdm.auto import tqdm
from nltk import sent_tokenize
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer


def load_model(path: str) -> LogisticRegression:
    with open(path, "rb") as f:
        return pickle.load(f)


def vectorize(text: str, model: SentenceTransformer) -> np.array:
    sentences = sent_tokenize(text)
    sentences_embeddings = model.encode(sentences)
    return sentences_embeddings.mean(axis=0)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return [vectorize(text, model).tolist() for text in tqdm(texts, desc="vectorizing texts")]


def predict(model: LogisticRegression, texts: list[str]) -> list[str]:
    vectors = embed_texts(texts)
    return model.predict(vectors).tolist()


def main():
    model_path = "model.pkl"
    texts = ["hello world", "how are you?"]
    model = load_model(model_path)
    labels = predict(model, texts)
    print(labels)


if __name__ == "__main__":
    main()
