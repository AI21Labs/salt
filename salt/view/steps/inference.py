import os
import pickle
import pandas as pd
import streamlit as st
from pathlib import Path
from salt.constants import VECTOR, TEXT
from salt.logic.embeddings import embed_texts
from salt.view.file_selector import file_selector
from salt.logic.classifier import SingleLabelClassifier
from salt.view.utils import get_project_state_if_has_classes

SINGLE = "🔤 Insert text"
FILE = "📃 Upload texts file"
CODE = "🧑🏻‍💻 From code"

INFER_FILE_KEY = "infer_file"
INFER_COL_KEY = "infer_col"
PROJECT_NAME_KEY = "project_name"


def get_filename(filepath: str) -> str:
    return os.path.basename(filepath)


def get_filename_wo_extension(filepath: str) -> str:
    return get_filename(filepath).split(".")[0]


def inference_step():
    project = get_project_state_if_has_classes()
    if not project:
        return

    inference_method = st.radio(
        label="Choose how to run the inference",
        options=(SINGLE, FILE, CODE),
        horizontal=True,
    )

    if inference_method == CODE:
        if project.al.is_multilabel:
            st.error("Sorry, this option is currently not supported for multi-label classifiers.")
            return

        project.al.fit()
        classifier: SingleLabelClassifier = project.al.model
        st.download_button("Export model", data=pickle.dumps(classifier.model), file_name="model.pkl")
        script_path = Path(__file__).parents[2].joinpath("resources/thin_classifier.py")
        with st.expander("Show code sample"):
            st.code(script_path.read_text(), language="python")
        return

    if inference_method == SINGLE:
        text = st.text_input("Insert text", label_visibility="collapsed")
        if not text:
            return
        df = pd.DataFrame({TEXT: [text], VECTOR: embed_texts([text])})
        filepath = "single"
        submitted = True

    else:  # inference_method == FILE:
        file_data = file_selector("inference")
        if file_data is None:
            file_data = st.session_state.get(INFER_FILE_KEY)
            if file_data is None:
                return
            df, filepath = file_data
            with st.form("existing_infer_file"):
                col1, col2 = st.columns([3, 1])
                col1.markdown(f"**File**: {get_filename(filepath)} , **Column**: {st.session_state[INFER_COL_KEY]}")
                submitted = col2.form_submit_button("Update predictions")
            if not submitted:
                st.dataframe(df.drop(columns={VECTOR}))
        else:
            df, filepath = file_data
            with st.form("new_infer_file"):
                col1, col2, col3 = st.columns([1, 2, 1])
                col1.markdown("Select column")
                column = col2.selectbox(
                    label=INFER_COL_KEY,
                    label_visibility="collapsed",
                    options=df.columns.tolist(),
                )
                submitted = col3.form_submit_button("Predict")
                if submitted:
                    df[VECTOR] = embed_texts(df[column].to_list())
                    st.session_state[INFER_FILE_KEY] = file_data
                    st.session_state[INFER_COL_KEY] = column
                else:
                    return

    if submitted:
        with st.spinner("Running..."):
            project.al.fit()
            project.al.predict_and_update(df)
        st.session_state[PROJECT_NAME_KEY] = project.name
        st.dataframe(df.drop(columns={VECTOR}))

    st.download_button(
        "Download",
        df.drop(columns={VECTOR}).to_csv(index=False),
        f"{st.session_state[PROJECT_NAME_KEY]}_inference_{get_filename_wo_extension(filepath)}.csv",
        "text/csv",
    )
