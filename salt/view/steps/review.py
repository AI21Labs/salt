import pandas as pd
import streamlit as st
from salt.logic.utils import get_prob_col
from salt.logic.project import SaltProject
from salt.logic.filter import FilterParams
from salt.view.utils import get_project_state, get_counts_df
from salt.constants import NA, SKIP, LABEL, PRED, CLUSTER, EDITED_DF_KEY, ALL

FILTER_PARAMS_KEY = "filter_params"
LABEL_FILTER_KEY = "label_filter"
PRED_FILTER_KEY = "pred_filter"
CLUSTER_FILTER_KEY = "cluster_filter"
QUERY_FILTER_KEY = "query_filter"
USE_SEMANTICS_FILTER_KEY = "use_semantics"
USE_REGEX_FILTER_KEY = "use_regex"

SINGLE_LABEL = "Single-label 📎️"
MULTI_LABEL = "Multi-label 🖇️️"


def get_df(project: SaltProject, filter_params: dict) -> pd.DataFrame:
    params = FilterParams()
    if filter_params[LABEL_FILTER_KEY] != ALL:
        params.label = filter_params[LABEL_FILTER_KEY]
    if filter_params[PRED_FILTER_KEY] != ALL:
        params.pred = filter_params[PRED_FILTER_KEY]
    if filter_params[CLUSTER_FILTER_KEY] != ALL:
        params.cluster = filter_params[CLUSTER_FILTER_KEY]
    params.query = filter_params[QUERY_FILTER_KEY]
    params.use_regex = filter_params.get(USE_REGEX_FILTER_KEY, True)
    params.use_semantics = filter_params.get(USE_SEMANTICS_FILTER_KEY, False)
    return project.get_data(params)


def review_step():
    project = get_project_state()
    if not project:
        return
    st.sidebar.header(project.name)
    al = project.al
    filter_params = {}

    col1, col2, col3, col4, col5, col6 = st.columns([0.6, 2.2, 1, 2.2, 0.7, 1.2])
    with col1:
        st.markdown("Label")
        st.markdown("###")
    with col2:
        filter_params[LABEL_FILTER_KEY] = st.selectbox(
            label=LABEL,
            options=[ALL] + al.labels + [SKIP, NA],
            label_visibility="collapsed",
        )
    with col3:
        st.markdown("Prediction")
        st.markdown("###")
    with col4:
        valid_pred_labels = [label for label in al.labels if get_prob_col(label) in project.df.columns]
        filter_params[PRED_FILTER_KEY] = st.selectbox(
            label=PRED, options=[ALL] + valid_pred_labels, label_visibility="collapsed"
        )
    with col5:
        st.markdown("Cluster")
        st.markdown("###")
    with col6:
        filter_params[CLUSTER_FILTER_KEY] = st.selectbox(
            label=CLUSTER,
            options=[ALL] + list(range(1, project.clusters.num_clusters + 1)),
            label_visibility="collapsed",
        )
    col7, col8, col9, col10, col11, col12 = st.columns([1.1, 5.5, 0.3, 1, 0.3, 0.8])
    with col7:
        st.markdown("Search 🔎")
        st.markdown("###")
    with col8:
        filter_params[QUERY_FILTER_KEY] = st.text_input("Search term", label_visibility="collapsed")
    with col10:
        st.markdown("Semantic")
        st.markdown("###")
    with col9:
        filter_params[USE_SEMANTICS_FILTER_KEY] = st.checkbox("Semantic", label_visibility="collapsed")
    if not filter_params[USE_SEMANTICS_FILTER_KEY]:
        with col12:
            st.markdown("Regex")
            st.markdown("###")
        with col11:
            filter_params[USE_REGEX_FILTER_KEY] = st.checkbox("Regex", label_visibility="collapsed")

    if filter_params != st.session_state.get(FILTER_PARAMS_KEY) or EDITED_DF_KEY not in st.session_state:
        st.session_state[FILTER_PARAMS_KEY] = filter_params
        df = get_df(project, filter_params)
        st.session_state[EDITED_DF_KEY] = df
        edited_df = st.data_editor(df, disabled=[col for col in df.columns if col != LABEL])
    else:
        edited_df = st.data_editor(
            st.session_state[EDITED_DF_KEY],
            disabled=[col for col in st.session_state[EDITED_DF_KEY].columns if col != LABEL],
        )
        df = get_df(project, filter_params)
        if not edited_df.equals(df):
            al.set_labels(edited_df)

    st.sidebar.markdown(
        MULTI_LABEL if al.is_multilabel else SINGLE_LABEL,
        help="Turn into Multi-label mode by adding a label with comma-separated classes (e.g. A,B)",
    )
    st.sidebar.dataframe(get_counts_df(edited_df))

    col3, col4 = st.sidebar.columns([1, 2])
    with col3:
        if st.button("Backup"):
            with st.spinner("Saving..."):
                project.dump_state()
    with col4:
        st.download_button(
            "Download",
            project.get_data().to_csv(index=False),
            project.state_filename,
            "text/csv",
        )
