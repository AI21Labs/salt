import math
import numpy as np
import pandas as pd
import streamlit as st
from salt.logic.clusters import DistanceType
from salt.view.utils import get_project_state
from salt.constants import EDITED_DF_KEY, CLUSTER, TEXT, MEAN_DISTANCE

CLUSTER_INDEX_KEY = "cluster_index"

EXPLICIT = "Explicit"
BY_DISTANCE = "By Distance"
OVERVIEW = "🗄️ Overview"
BY_CLUSTER = "🗃️ By Cluster"


def clusters_step():
    project = get_project_state()
    if not project:
        return

    if EDITED_DF_KEY in st.session_state:
        st.session_state.pop(EDITED_DF_KEY)

    clusters = project.clusters
    if clusters.num_clusters == 1:
        st.session_state[CLUSTER_INDEX_KEY] = 1

    st.sidebar.header(project.name)
    distance_type = st.sidebar.radio(
        label="Similarity type",
        options=[t.value for t in DistanceType],
        horizontal=True,
        help=(
            "**Lexical**: by exact phrases (to find patterns)\n\n" "**Semantic**: by similar meaning (to find topics)"
        ),
    )

    method = st.sidebar.radio(label="Number of clusters", options=(EXPLICIT, BY_DISTANCE), horizontal=True)
    if method == EXPLICIT:
        default_num_clusters = math.ceil(
            clusters.num_examples / 2 if distance_type == DistanceType.LEXICAL.value else clusters.num_examples / 100
        )
        num_clusters = st.sidebar.number_input(
            "Num clusters",
            label_visibility="collapsed",
            min_value=1,
            max_value=clusters.num_examples,
            value=default_num_clusters,
        )
        distance_threshold = None
    else:
        distance_threshold = st.sidebar.slider(
            "Distance threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help=(
                "The maximum allowed distance between two items in a cluster.\n\n"
                "**Small** threshold will return many small & tight clusters.\n\n"
                "**Large** threshold will return few big & loose clusters."
            ),
        )
        num_clusters = None

    if st.sidebar.button("Run clustering"):
        with st.spinner("Building clusters..."):
            clusters.run(
                num_clusters=num_clusters,
                distance_threshold=distance_threshold,
                distance_type=DistanceType(distance_type),
            )
            project.update_clusters(clusters.df)
        st.session_state[CLUSTER_INDEX_KEY] = 1

    df_all = (
        clusters.get_data()
        .groupby(CLUSTER)
        .agg(
            cluster=pd.NamedAgg(column=CLUSTER, aggfunc=lambda ids: ids.iloc[0]),
            example=pd.NamedAgg(column=TEXT, aggfunc=lambda texts: texts.iloc[0]),
            size=pd.NamedAgg(column=TEXT, aggfunc="count"),
            cohesion=pd.NamedAgg(
                column=MEAN_DISTANCE,
                aggfunc=lambda distances: round(1 - np.mean(distances), 2),
            ),
        )
        .set_index("cluster", drop=True)
    )

    view_mode = st.radio(
        "View mode",
        options=[OVERVIEW, BY_CLUSTER],
        label_visibility="collapsed",
        horizontal=True,
    )
    if view_mode == OVERVIEW:
        st.dataframe(df_all, use_container_width=True)
    else:  # view_mode == BY_CLUSTER
        col1, col2, _ = st.columns([0.7, 1.4, 3])
        with col1:
            st.markdown("cluster id:")
        with col2:
            st.number_input(
                "cluster id",
                min_value=1,
                max_value=clusters.num_clusters,
                step=1,
                label_visibility="collapsed",
                key=CLUSTER_INDEX_KEY,
            )
        st.dataframe(
            clusters.get_data(st.session_state[CLUSTER_INDEX_KEY]).reset_index(drop=True),
            use_container_width=True,
        )
    st.sidebar.download_button(
        "Download",
        clusters.get_data().to_csv(index=False),
        f"{project.name}_clusters-{clusters.distance_type.value.lower()}-{clusters.num_clusters}.csv",
        "text/csv",
    )
