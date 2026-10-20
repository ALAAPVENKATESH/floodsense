"""Project summary, pipeline, and the scope limits of the coursework model."""

from __future__ import annotations

import streamlit as st

import config
from app.ui.components import callout, feature_pill, pipeline, render, section_header
from app.views._frame import header


def page() -> None:
    """About and limitations."""
    header("About and limitations", "Machine Learning (BIC703), semester VII, group G2.")
    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True):
            render(section_header("FloodSense", "Urban flood-risk intelligence for a coursework classifier."))
            st.markdown(
                "The app estimates **LOW**, **MODERATE**, or **HIGH** flood risk for one urban location "
                "from rainfall, terrain, drainage, land cover, and a historical flood indicator. "
                "The decision tree and the random forest are written in this repository with NumPy."
            )
            render(section_header("Pipeline"))
            render(pipeline())
    with right:
        with st.container(border=True):
            render(section_header("Stack"))
            render(
                "".join(
                    feature_pill(name)
                    for name in (
                        "Python",
                        "NumPy",
                        "Pandas",
                        "Matplotlib",
                        "Plotly",
                        "Streamlit",
                        "Self-implemented random forest",
                    )
                )
            )
            render(section_header("Course"))
            st.markdown(
                "Subject **Machine Learning (BIC703)**. Semester VII. Group **G2**. "
                "Technique: random forest classification. "
                "scikit-learn is used in the tests and the benchmark script only."
            )
    st.write("")
    with st.container(border=True):
        render(section_header("Limitations"))
        st.markdown(
            """
- This is a coursework classifier. It is not a flood warning system.
- A real deployment would need validated operational data, spatial and temporal checks, uncertainty estimates, and review by people who work on floods.
- When the banner is up, the scores are not a surveyed flood record.
- Feature importance is model behavior, not a cause of flooding.
- Live rainfall, sensors, automated alerts, and a mobile app are out of scope.
            """
        )
        render(callout(config.CAUSATION_NOTE))
        render(callout(config.DISCLAIMER))
