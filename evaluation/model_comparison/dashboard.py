# Streamlit dashboard for comparing VLM model benchmark results from run_comparison.py.

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

#? ---
#? PATHS
#? ---

RESULTS_DIR = Path(__file__).parent / "results"
EVAL_DIR = Path(__file__).parent.parent

#? ---
#? DATA LOADING
#? ---

# Returns a sorted list of result JSON files in the results/ folder
def _list_result_files():
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("comparison_*.json"), reverse=True)

# Loads a JSON result file and returns (run_id, list of result dicts)
def _load_file(path: Path):
    data = json.loads(path.read_text())
    return data.get("run_id", path.stem), data.get("results", [])

# Flattens result records into a DataFrame, expanding parsed fields
def _to_dataframe(results: list):
    rows = []
    for r in results:
        parsed = r.get("parsed", {})
        rows.append({
            "model":        r["model"],
            "strategy":     r["strategy"],
            "image":        r["image"],
            "elapsed_s":    r["elapsed_s"],
            "success":      r["success"],
            "well_formed":  r["well_formed"],
            "action":       r.get("action") or "—",
            "timing_note":  r.get("timing_note", ""),
            "n_runs":       r.get("n_runs", 1),
            "obstacle_type": parsed.get("obstacle_type", ""),
            "movement":     parsed.get("movement", ""),
            "if_moving":    parsed.get("if_moving", ""),
            "confidence":   parsed.get("confidence", ""),
            "description":  parsed.get("description", ""),
            "reasoning":    parsed.get("reasoning", ""),
            "raw":          r.get("raw") or "",
            "prompt":       r.get("prompt") or "",
        })
    return pd.DataFrame(rows)


#? ---
#? COLOUR HELPERS
#? ---

_ACTION_COLOURS = {
    "STOP":       "#e74c3c",
    "CONTINUE":   "#2ecc71",
    "TURN_LEFT":  "#e67e22",
    "TURN_RIGHT": "#e67e22",
}
_CONF_COLOURS = {"high": "#2ecc71", "medium": "#f39c12", "low": "#e74c3c"}

# Wraps text in a coloured badge span
def _badge(text: str, colour: str):
    return f'<span style="background:{colour};color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8em;font-weight:bold">{text}</span>'

# Returns an action badge HTML string
def _action_badge(action: str):
    colour = _ACTION_COLOURS.get(action.upper(), "#888")
    return _badge(action, colour)

# Returns a confidence badge HTML string
def _conf_badge(conf: str):
    colour = _CONF_COLOURS.get(conf.lower(), "#888")
    return _badge(conf, colour)


#? ---
#? PAGE: OVERVIEW
#? Summary table and timing chart across all models and strategies.
#? ---

def _page_overview(df: pd.DataFrame):
    st.subheader("Timing & Actions")

    models = df["model"].unique().tolist()
    strategies = df["strategy"].unique().tolist()

    col_model, col_strat = st.columns(2)
    sel_models = col_model.multiselect("Models", models, default=models)
    sel_strats = col_strat.multiselect("Strategies", strategies, default=strategies)

    filtered = df[df["model"].isin(sel_models) & df["strategy"].isin(sel_strats)]

    if filtered.empty:
        st.info("No data matches the current filters.")
        return

    # Timing bar chart
    chart_data = filtered[["model", "strategy", "elapsed_s"]].dropna()
    chart_data["label"] = chart_data["model"] + " / " + chart_data["strategy"]
    st.bar_chart(chart_data.set_index("label")["elapsed_s"], use_container_width=True)
    timing_note = filtered["timing_note"].dropna().unique()
    if timing_note.size > 0:
        st.caption(f"Timing note: {timing_note[0]}")

    st.divider()

    # Summary table with colour-coded action column
    st.subheader("Results at a glance")
    for _, row in filtered.iterrows():
        c1, c2, c3, c4, c5 = st.columns([3, 2, 1.5, 1.5, 1.5])
        c1.markdown(f"**{row['model']}**  `{row['strategy']}`")
        c2.caption(row["image"])
        c3.markdown(_action_badge(row["action"]), unsafe_allow_html=True)
        c4.markdown(_conf_badge(row["confidence"]) if row["confidence"] else "—", unsafe_allow_html=True)
        elapsed = f"{row['elapsed_s']:.1f}s" if row["elapsed_s"] is not None else "—"
        wf_icon = "✅" if row["well_formed"] else ("❌" if row["success"] else "⚠️")
        c5.caption(f"{elapsed}  {wf_icon}")


#? ---
#? PAGE: RESPONSE COMPARISON
#? Side-by-side cards for each model × strategy combination.
#? ---

def _page_responses(df: pd.DataFrame):
    st.subheader("Side-by-side response comparison")

    images = df["image"].unique().tolist()
    sel_image = st.selectbox("Image", images)
    img_df = df[df["image"] == sel_image]

    strategies = img_df["strategy"].unique().tolist()
    sel_strategy = st.selectbox("Strategy", strategies)
    strat_df = img_df[img_df["strategy"] == sel_strategy]

    if strat_df.empty:
        st.info("No results for this combination.")
        return

    # Show the image if it exists on disk
    image_path = EVAL_DIR / "recordings" / "images" / sel_image
    if not image_path.exists():
        # Try finding it by name alone
        matches = list(EVAL_DIR.rglob(sel_image))
        if matches:
            image_path = matches[0]
    if image_path.exists():
        st.image(str(image_path), use_container_width=True)

    st.divider()

    cols = st.columns(len(strat_df))
    for col, (_, row) in zip(cols, strat_df.iterrows()):
        with col:
            st.markdown(f"### {row['model']}")
            st.caption(f"`{row['strategy']}`")
            elapsed = f"{row['elapsed_s']:.1f}s" if row["elapsed_s"] is not None else "—"
            wf = "✅ all fields" if row["well_formed"] else ("❌ incomplete" if row["success"] else "⚠️ error")
            st.caption(f"⏱ {elapsed}  |  {wf}")

            if row["action"] and row["action"] != "—":
                st.markdown(_action_badge(row["action"]), unsafe_allow_html=True)

            st.divider()

            if row["description"]:
                st.markdown(f"**Description**  \n{row['description']}")
            if row["obstacle_type"]:
                st.markdown(f"**Obstacle:** `{row['obstacle_type']}`")
            if row["movement"]:
                st.markdown(f"**Movement:** `{row['movement']}`")
            if row["if_moving"]:
                st.markdown(f"**If moving:** `{row['if_moving']}`")
            if row["confidence"]:
                st.markdown(_conf_badge(row["confidence"]), unsafe_allow_html=True)
            if row["reasoning"]:
                st.caption(f"Reasoning: {row['reasoning']}")

            with st.expander("Raw response"):
                st.code(row["raw"] or "(no response)", language=None)


#? ---
#? PAGE: PROMPTS
#? Shows the full prompt text used for each strategy.
#? ---

def _page_prompts(df: pd.DataFrame):
    st.subheader("Prompts used")
    st.caption("These are the exact prompts sent to the VLM for each strategy.")

    strategies = df["strategy"].unique().tolist()
    for strategy in strategies:
        rows = df[df["strategy"] == strategy]
        prompt = rows["prompt"].dropna().iloc[0] if not rows["prompt"].dropna().empty else ""
        with st.expander(f"`{strategy}`", expanded=True):
            st.code(prompt, language=None)


#? ---
#? PAGE: RAW DATA
#? Filterable full table of all results.
#? ---

def _page_raw(df: pd.DataFrame):
    st.subheader("All results")

    display_cols = ["model", "strategy", "image", "elapsed_s", "success",
                    "well_formed", "action", "obstacle_type", "movement",
                    "if_moving", "confidence", "description"]
    st.dataframe(df[display_cols], use_container_width=True)

    st.download_button(
        label="Download as CSV",
        data=df[display_cols].to_csv(index=False),
        file_name="comparison_results.csv",
        mime="text/csv",
    )


#? ---
#? APP ENTRY POINT
#? File selector in sidebar, tab navigation in main area.
#? ---

def main():
    st.set_page_config(page_title="VLM Benchmark", layout="wide")
    st.title("VLM Benchmark Dashboard")

    result_files = _list_result_files()
    if not result_files:
        st.warning(f"No result files found in `{RESULTS_DIR}`. Run `run_comparison.py` first.")
        return

    with st.sidebar:
        st.header("Result files")
        file_labels = {f.name: f for f in result_files}
        selected_names = st.multiselect(
            "Select runs to compare",
            list(file_labels.keys()),
            default=[list(file_labels.keys())[0]],
        )
        if not selected_names:
            st.info("Select at least one result file.")
            return

    all_results = []
    for name in selected_names:
        _, results = _load_file(file_labels[name])
        all_results.extend(results)

    if not all_results:
        st.error("Selected files contain no results.")
        return

    df = _to_dataframe(all_results)

    with st.sidebar:
        st.divider()
        st.metric("Total results", len(df))
        st.metric("Models", df["model"].nunique())
        st.metric("Strategies", df["strategy"].nunique())
        st.metric("Images", df["image"].nunique())

    tab_overview, tab_responses, tab_prompts, tab_raw = st.tabs([
        "Overview", "Response Comparison", "Prompts", "Raw Data"
    ])

    with tab_overview:
        _page_overview(df)

    with tab_responses:
        _page_responses(df)

    with tab_prompts:
        _page_prompts(df)

    with tab_raw:
        _page_raw(df)


if __name__ == "__main__":
    main()
