import hashlib
import os

import streamlit as st

from pyologger.utils.folder_manager import load_configuration


st.set_page_config(page_title="Pyologger", layout="wide")

logo_path = "/Users/jessiekb/Documents/GitHub/EcoViz_DiveDB/pyologger/assets/pyologger_logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=220)

st.markdown(
    """
    <h1 style="margin-bottom:0.15rem;">pyologger</h1>
    <p style="margin-top:0; font-size:1.05rem; color:#5a6772;">
        visu-analyze multi-sensor ecophysiological data
    </p>
    """,
    unsafe_allow_html=True,
)
st.caption("Select a dataset card to set the default dataset across app pages.")


def _list_datasets(data_dir):
    return sorted(
        [
            d
            for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith("00_")
        ]
    )


def _deployment_count(data_dir, dataset_name):
    ds_dir = os.path.join(data_dir, dataset_name)
    return len(
        [
            d
            for d in os.listdir(ds_dir)
            if os.path.isdir(os.path.join(ds_dir, d)) and not d.startswith("00_")
        ]
    )


def _stable_card_color(name):
    palette = [
        "#E8F4F8",
        "#FBEAD1",
        "#E7F5E8",
        "#F7EAF6",
        "#EAF0FB",
        "#FDECEC",
        "#EEF7F1",
        "#FFF4DE",
    ]
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(palette)
    return palette[idx]


config, data_dir, _, _ = load_configuration()
datasets = _list_datasets(data_dir)
if not datasets:
    st.error("No datasets found.")
    st.stop()

current_default = st.session_state.get("preferred_dataset_selection")
if current_default not in datasets:
    current_default = datasets[0]
    st.session_state["preferred_dataset_selection"] = current_default

st.info(f"Current default dataset: `{current_default}`")

n_cols = 3
for i in range(0, len(datasets), n_cols):
    cols = st.columns(n_cols)
    for j, ds in enumerate(datasets[i : i + n_cols]):
        with cols[j]:
            bg = _stable_card_color(ds)
            st.markdown(
                f"""
                <div style="background:{bg}; border-radius:10px; padding:0.8rem 1rem; border:1px solid #d5dde6;">
                    <div style="font-size:1.05rem; font-weight:700; margin-bottom:0.35rem;">{ds}</div>
                    <div style="font-size:0.92rem; color:#3f4b57;">Deployments: {_deployment_count(data_dir, ds)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Use This Dataset", key=f"use_ds_{ds}"):
                st.session_state["preferred_dataset_selection"] = ds
                st.session_state["dataset_selection"] = ds
                st.success(f"Default dataset set to: {ds}")

st.caption("Use the sidebar page navigation to open processing pages.")
