"""Streamlit app for inspecting generated train/val/test topology samples."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
import yaml
from pathlib import Path

from training_data.visualize import (
    build_pyvis_graph,
    compute_sample_stats,
    compute_split_stats,
    load_split_samples,
)

st.set_page_config(page_title="HGT Topology Visualizer", layout="wide")
st.title("HGT Topology Visualizer")

# ── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")
config_path = st.sidebar.text_input("Config YAML path", value="configs/config.yaml")

config: dict | None = None
dataset_paths: dict[str, str] = {}

if config_path and Path(config_path).exists():
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    dataset_paths = config.get("dataset_paths", {})
    st.sidebar.success(f"Loaded config from {config_path}")
else:
    st.sidebar.warning("Config file not found. Enter a valid path above.")

available_splits = [
    name for name in ("train", "val", "test")
    if name in dataset_paths and Path(dataset_paths[name]).exists()
]

if not available_splits:
    st.info(
        "No generated data found. Run the pipeline first:\n\n"
        "```\npython main.py --config configs/config.yaml --mode train\n```\n\n"
        "Or generate data only by running the export step."
    )
    st.stop()

split_name = st.sidebar.selectbox("Split", available_splits)


# ── Load data ────────────────────────────────────────────────────────────
@st.cache_data
def _load_samples(path: str) -> list[dict]:
    return load_split_samples(path)


samples = _load_samples(dataset_paths[split_name])

if not samples:
    st.warning(f"No samples found in {dataset_paths[split_name]}")
    st.stop()

st.sidebar.markdown(f"**{len(samples)}** samples in `{split_name}`")

sample_idx = st.sidebar.slider(
    "Sample index",
    min_value=0,
    max_value=len(samples) - 1,
    value=0,
)

sample = samples[sample_idx]

# ── Split-level stats ────────────────────────────────────────────────────
with st.expander("Split-level statistics", expanded=False):
    split_stats = compute_split_stats(samples)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total samples", split_stats["total_samples"])
    col2.metric("Avg nodes/sample", split_stats["nodes"]["mean"])
    col3.metric("Avg edges/sample", split_stats["edges"]["mean"])

    col4, col5, col6 = st.columns(3)
    col4.metric("Avg sites/sample", split_stats["sites"]["mean"])
    col5.metric("Positive label ratio (mean)", f"{split_stats['positive_ratio']['mean']:.3f}")
    col6.metric("Trainable positive ratio (mean)", f"{split_stats['trainable_positive_ratio']['mean']:.3f}")

    st.markdown("**Fault mode distribution:**")
    st.json(split_stats["fault_mode_distribution"])

# ── Sample-level stats ───────────────────────────────────────────────────
st.subheader(f"Sample: {sample.get('sample_id', sample_idx)}")

stats = compute_sample_stats(sample)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Sites", stats["num_sites"])
col_b.metric("Nodes", stats["num_nodes"])
col_c.metric("Edges", stats["num_edges"])
col_d.metric("Seed", stats["seed"])

col_e, col_f, col_g = st.columns(3)
col_e.metric("Alarm entities", stats["total_alarm_entities"])
col_f.metric("Positive labels", stats["positive_alarm_entities"])
col_g.metric("Trainable positive", f"{stats['trainable_positive']}/{stats['trainable_alarm_entities']}")

with st.expander("Sample details", expanded=False):
    det1, det2 = st.columns(2)
    with det1:
        st.markdown("**Fault sites & modes:**")
        for site in stats["fault_sites"]:
            mode = stats["fault_modes"].get(site, "unknown")
            st.markdown(f"- `{site}`: {mode}")
    with det2:
        st.markdown(f"**AN sites:** {', '.join(f'`{s}`' for s in stats['an_sites']) or 'none'}")
        st.markdown(f"**Noise sites:** {', '.join(f'`{s}`' for s in stats['noise_sites']) or 'none'}")
        st.markdown(f"**Inactive NEs:** {len(stats['inactive_ne_ids'])}")
        st.markdown(f"**Blocked edges:** {len(stats['blocked_edge_pairs'])}")

# ── Interactive graph ────────────────────────────────────────────────────
st.subheader("Topology Graph")

st.markdown(
    "Node colors: "
    ":blue[phy_site] | "
    ":green[router] | "
    ":orange[wl_station] "
    "— Red border = outage — Star = fault anchor — Diamond = AN"
)

net = build_pyvis_graph(sample, height="600px")
html = net.generate_html()
components.html(html, height=620, scrolling=True)
