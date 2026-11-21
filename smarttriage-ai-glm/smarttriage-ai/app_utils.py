"""Shared Streamlit helpers: cached model loader, session-state init, styling.

Kept outside ``pages/`` so it is not treated as a page itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable when Streamlit runs from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from config import settings
from services.ticket_service import get_backend


def init_state() -> None:
    """Initialise session state once."""
    defaults = {
        "predictions": None,        # latest predictions DataFrame
        "input_df": None,           # last uploaded/loaded tickets DataFrame
        "processing_status": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


@st.cache_resource(show_spinner="Loading zero-shot classification model...")
def cached_backend():
    """Load the active backend once per process (Streamlit resource cache).

    If the model cannot be downloaded (no network on Streamlit Community Cloud,
    disk limits, etc.) we return the backend object anyway; its ``load_error``
    is surfaced to the UI and every ticket is routed to human review rather
    than crashing the app.
    """
    return get_backend()


def get_predictions() -> pd.DataFrame | None:
    init_state()
    return st.session_state.predictions


def set_predictions(df: pd.DataFrame | None) -> None:
    init_state()
    st.session_state.predictions = df


def apply_theme() -> None:
    """Navy/white SaaS theme via inline CSS."""
    st.markdown(
        f"""
        <style>
        /* Base app and layout */
        .stApp {{
            background-color: #FFFFFF !important;
            color: #0A2540 !important;
        }}

        /* Force high-contrast dark text for all text and headers */
        .stApp p, .stApp span, .stApp label, .stApp li,
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stApp table, .stApp th, .stApp td, .stApp caption {{
            color: #0A2540 !important;
        }}

        /* Streamlit metrics */
        [data-testid="stMetricValue"] {{
            color: #0A2540 !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: #475569 !important;
        }}

        /* Sidebar styling - crisp light background with dark readable text */
        section[data-testid="stSidebar"],
        [data-testid="stSidebarContent"],
        [data-testid="stSidebar"] > div:first-child {{
            background-color: #F8FAFC !important;
            border-right: 1px solid #E2E8F0;
        }}

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] a,
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] caption {{
            color: #0A2540 !important;
        }}

        /* Navigation links inside sidebar */
        [data-testid="stSidebarNav"] span,
        [data-testid="stSidebarNav"] a,
        [data-testid="stSidebarNavItems"] span,
        [data-testid="stSidebarNavItems"] a {{
            color: #0A2540 !important;
        }}

        /* Subtitles and captions */
        .stCaption, [data-testid="stCaptionContainer"] {{
            color: #64748B !important;
        }}

        /* Custom UI classes */
        .bigtitle {{
            color: {settings.COLORS['navy']} !important;
            font-weight: 800;
            font-size: 1.6rem;
            letter-spacing: -0.5px;
        }}
        .subtitle {{
            color: {settings.COLORS['grey']} !important;
            font-size: 0.95rem;
        }}
        .metric-card {{
            background: #F8FAFC !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px;
            padding: 14px 16px;
        }}

        /* Status pills (badges maintain crisp white text on solid backgrounds) */
        .pill {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            color: #FFFFFF !important;
        }}
        .pill-green  {{ background: {settings.COLORS['green']} !important; color: #FFFFFF !important; }}
        .pill-yellow {{ background: {settings.COLORS['yellow']} !important; color: #FFFFFF !important; }}
        .pill-red    {{ background: {settings.COLORS['red']} !important; color: #FFFFFF !important; }}
        .pill-navy   {{ background: {settings.COLORS['navy']} !important; color: #FFFFFF !important; }}

        /* Primary action button: white text */
        button[kind="primary"], button[data-testid="baseButton-primary"] {{
            color: #FFFFFF !important;
        }}
        button[kind="primary"] * {{
            color: #FFFFFF !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(title: str, subtitle: str = "") -> None:
    apply_theme()
    st.markdown(f'<div class="bigtitle">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.markdown("---")


def priority_pill(priority: str) -> str:
    color = {"Urgent": "pill-red", "High": "pill-yellow",
             "Medium": "pill-navy", "Low": "pill-green"}.get(priority, "pill-navy")
    return f'<span class="pill {color}">{priority or "—"}</span>'


def load_default_tickets() -> pd.DataFrame:
    """Load the bundled placeholder dataset (or the user's real one)."""
    if settings.TICKETS_CSV.exists():
        return pd.read_csv(settings.TICKETS_CSV)
    return pd.DataFrame(columns=["id", "subject", "body"])
