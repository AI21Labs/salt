import streamlit as st
from pathlib import Path


def about_step():
    readme_path = Path(__file__).parents[3].joinpath("README.md")
    st.markdown(readme_path.read_text())
