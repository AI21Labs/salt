import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from salt.logic.utils import get_str_from_labels
from salt.constants import DUMP_INTERVAL, EDITED_DF_KEY, SKIP
from salt.view.utils import get_project_state_if_has_classes, get_counts_df


def labeling_step():
    project = get_project_state_if_has_classes()
    if not project:
        return

    if EDITED_DF_KEY in st.session_state:
        st.session_state.pop(EDITED_DF_KEY)

    st_text = st.empty()
    al = project.al
    if al.all_labeled:
        st.info("All texts are labeled!")
        return

    st_text.text_area(label="example to annotate", value=al.next_example, label_visibility="collapsed")

    labels = al.get_ann_options()

    buttons_per_row = 5
    label2check = {}
    annotation = None
    is_multilabel = al.is_multilabel
    with st.form("labels", clear_on_submit=True):
        for label_ind in range(0, len(labels), buttons_per_row):
            row_labels = labels[label_ind : label_ind + buttons_per_row]
            for label, col in zip(row_labels, st.columns(buttons_per_row)):
                click = col.form_submit_button(label)
                if click:
                    annotation = label
                if is_multilabel and label != SKIP:
                    label2check[label] = col.checkbox(label=label, label_visibility="collapsed")
        if is_multilabel and st.form_submit_button("Submit All Checked"):
            annotation = get_str_from_labels([label for label, checked in label2check.items() if checked])

    if annotation:
        with st.spinner("Updating model..."):
            al.step(annotation)
        if al.all_labeled:
            st.info("All texts are labeled!")
            return
        st_text.text_area(
            label="example to annotate",
            value=al.next_example,
            label_visibility="collapsed",
        )

        st.markdown("")
        st.markdown("###### Class Distribution")
        st.dataframe(get_counts_df(al.df).T)

        if al.num_anns % DUMP_INTERVAL == 0:
            with st.spinner("Auto-save..."):
                project.dump_state()

        df_history = al.update_history_and_get_change_df()
        if df_history is None:
            return

        fig = px.line(
            df_history,
            x="num_labels",
            y="change_rate",
            color="class",
            markers=True,
            title="How Stable is my Classifier?",
        )
        fig.update_layout(
            shapes=[
                go.layout.Shape(
                    type="rect",
                    xref="paper",
                    yref="paper",
                    x0=0,
                    y0=0,
                    x1=1.0,
                    y1=1.0,
                    line={"width": 1, "color": "gray"},
                )
            ]
        )
        st.plotly_chart(fig, use_container_width=True)
