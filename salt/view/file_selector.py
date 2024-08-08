import os
import tempfile
import pandas as pd
import streamlit as st
from typing import Optional
from salt.utils import read_csv_or_jsonl

LOADED_DF_KEY = "loaded_df"
FILE_ID_KEY = "file_id"


def _load_file(path: str) -> pd.DataFrame:
    try:
        with st.spinner("Loading..."):
            return read_csv_or_jsonl(path)
    except FileNotFoundError:
        st.error("File not found.")


def file_selector(label: str) -> Optional[tuple[pd.DataFrame, str]]:
    def get_state_key(key):
        return f"{label}_{key}"

    uploaded_file = st.file_uploader(
        "Choose a file",
        label_visibility="collapsed",
        type=["csv", "jsonl"],
        key=get_state_key("uploaded_file"),
    )

    if not uploaded_file:
        st.session_state[get_state_key(FILE_ID_KEY)] = None
        return None

    if uploaded_file.name != st.session_state.get(get_state_key(FILE_ID_KEY)):  # new file uploaded
        tmp_dir = tempfile.TemporaryDirectory()
        filepath = os.path.join(tmp_dir.name, uploaded_file.name)
        with open(filepath, "w") as f:
            f.write(uploaded_file.getvalue().decode("utf-8"))
        st.session_state[get_state_key(LOADED_DF_KEY)] = _load_file(filepath)
        st.session_state[get_state_key(FILE_ID_KEY)] = uploaded_file.name

    return (
        st.session_state[get_state_key(LOADED_DF_KEY)],
        st.session_state[get_state_key(FILE_ID_KEY)],
    )
