import pandas as pd


def get_file_type(path: str) -> str:
    return path.split(".")[-1]


def read_csv_or_jsonl(path: str) -> pd.DataFrame:
    file_type = get_file_type(path)
    if file_type == "csv":
        return pd.read_csv(path)
    if file_type == "jsonl":
        return pd.read_json(path, lines=True)

    raise NotImplementedError("Only CSV and JSONL file types are supported.")
