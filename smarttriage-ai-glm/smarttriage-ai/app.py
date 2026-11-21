"""SmartTriage AI — Streamlit entry point.

Wires the five pages together with ``st.navigation``. Each page module exposes
a ``render()`` callable; the active backend is loaded once via the Streamlit
resource cache in ``app_utils.cached_backend``.
"""
from __future__ import annotations

import streamlit as st

import app_utils
from pages import (analytics, dashboard, human_review, model_information,
                   ticket_processing)

st.set_page_config(
    page_title="SmartTriage AI",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

app_utils.apply_theme()
app_utils.init_state()

pages = {
    "SmartTriage AI": [
        st.Page(dashboard.render, title="Overview", url_path="overview", default=True, icon="📊"),
        st.Page(ticket_processing.render, title="Ticket Processing", url_path="ticket-processing", icon="⚙️"),
        st.Page(human_review.render, title="Human Review Center", url_path="human-review", icon="✅"),
        st.Page(analytics.render, title="Analytics", url_path="analytics", icon="📈"),
        st.Page(model_information.render, title="Model Information", url_path="model-information", icon="🧠"),
    ]
}

nav = st.navigation(pages)
with st.sidebar:
    st.markdown(
        f"### 🧭 {app_utils.settings.APP_NAME}\n"
        f"<div style='color:#64748B;font-size:0.85rem'>"
        f"{app_utils.settings.APP_TAGLINE}</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    backend = app_utils.cached_backend()
    if hasattr(backend, "is_loaded") and backend.is_loaded:
        st.success("Zero-shot model loaded", icon="✅")
    elif hasattr(backend, "load_error") and backend.load_error:
        st.error("Model unavailable — reviews only", icon="⚠️")
    st.caption(f"Backend: `{app_utils.settings.ACTIVE_BACKEND}`")

nav.run()
