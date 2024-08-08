import streamlit as st
import extra_streamlit_components as stx
from salt.view.steps.setup import setup_step
from salt.view.steps.about import about_step
from salt.view.steps.review import review_step
from salt.view.steps.clusters import clusters_step
from salt.view.steps.labeling import labeling_step
from salt.view.steps.inference import inference_step


def main():
    st.set_page_config(page_title="SALT", page_icon="🧂")
    st.header(f"🧂 SALT 🧂 Simple Active-Learning Tool 🧂")
    step = stx.stepper_bar(
        [
            "Setup ⚙️️",
            "Clusters 🗄️",
            "Review 📖",
            "Labeling 🖊️️",
            "Inference 🔦",
            "About 📓",
        ],
        lock_sequence=False,
    )
    if step == 0:
        setup_step()

    elif step == 1:
        clusters_step()

    elif step == 2:
        review_step()

    elif step == 3:
        labeling_step()

    elif step == 4:
        inference_step()

    elif step == 5:
        about_step()


if __name__ == "__main__":
    main()
