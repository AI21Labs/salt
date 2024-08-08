import re
import streamlit as st
from pathlib import Path
from typing import Optional
from salt.logic.project import SaltProject
from salt.view.file_selector import file_selector
from salt.constants import PROJECTS_DIR, PROJECT_STATE_KEY, EDITED_DF_KEY

PROJECT_LIST_KEY = "project_list"

CREATE = "➕ Create"
LOAD = "📁 Load"


def get_type(filepath: str) -> str:
    return filepath.split(".")[-1]


def find_project_name_problems(name: str) -> Optional[str]:
    if not name:
        return "Project name cannot be empty."
    invalid_char = re.search(r"[^\w_-]", name)
    if invalid_char:
        return f"Project name cannot contain the character '{invalid_char.group()}'"
    if name in st.session_state[PROJECT_LIST_KEY]:
        return "Project name already exists."
    return None


def setup_step():
    if EDITED_DF_KEY in st.session_state:
        st.session_state.pop(EDITED_DF_KEY)

    if PROJECT_LIST_KEY not in st.session_state:
        with st.spinner("Loading..."):
            projects_dir = Path(PROJECTS_DIR)
            projects_dir.mkdir(parents=True, exist_ok=True)

            st.session_state[PROJECT_LIST_KEY] = sorted([p.name for p in projects_dir.iterdir() if p.is_dir()])

    create_or_load = st.radio(
        label="Create a new project or load an existing one",
        options=(CREATE, LOAD),
        horizontal=True,
    )

    if create_or_load == LOAD:
        selected_project = st.selectbox(label="Select project", options=st.session_state[PROJECT_LIST_KEY])
        if st.button("Load project"):
            with st.spinner("Loading project..."):
                st.session_state[PROJECT_STATE_KEY] = SaltProject.load(selected_project)
            st.write("Project loaded successfully! To proceed, click on one of the steps above.")
        return

    # CREATE
    file_data = file_selector("setup")
    if file_data is None:
        return
    df = file_data[0]

    with st.form("form"):
        col1, col2 = st.columns([1, 5.2])
        col1.markdown("Text column")
        text_column = col2.selectbox(label="text_col", label_visibility="collapsed", options=df.columns.tolist())

        col3, col4, col5 = st.columns([1.4, 5.5, 1.7])
        col3.markdown("Project name")
        project_name = col4.text_input(label="project_name", label_visibility="collapsed")
        submitted = col5.form_submit_button("Create project")

        with st.expander("Optional settings"):
            col6, col7 = st.columns([1, 5])
            col6.markdown("Label column")
            label_column = col7.selectbox(
                label="label_col",
                label_visibility="collapsed",
                options=[None] + df.columns.tolist(),
            )

            col8, col9 = st.columns([1, 5])
            col8.markdown("Base project")
            base_project_name = col9.selectbox(
                label="base_project",
                label_visibility="collapsed",
                options=[None] + st.session_state[PROJECT_LIST_KEY],
            )

    if submitted:
        name_problem = find_project_name_problems(project_name)
        if name_problem:
            st.error(name_problem)
            return
        with st.spinner("Creating project..."):
            st.session_state[PROJECT_STATE_KEY] = SaltProject.create(
                df, text_column, label_column, project_name, base_project_name
            )
        st.write("Project created successfully! To proceed, click on one of the steps above.")
