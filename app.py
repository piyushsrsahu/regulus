"""
Regulus web application.
"""
import os
import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src import PROJECT_ROOT
from src.preprocessing import load_light_curve, parse_light_curve_frame
from src.features import extract_features
from src.model import predict
from src.db import init_db, save_analysis, load_history, clear_history

# Unicode character codes for interface symbols
ICON_PLANET = "\U0001FA90"     # Ringed Planet
ICON_STAR = "\u2B50"           # Star
ICON_SPARKLE = "\u2728"        # Sparkles
ICON_ANALYZE = "\U0001F4CA"    # Chart
ICON_HISTORY = "\U0001F4DC"    # History Scroll
ICON_SEARCH = "\U0001F50D"     # Magnifying Glass

# Initialize local SQLite storage
init_db()

st.set_page_config(
    page_title="Regulus",
    page_icon=ICON_PLANET,
    layout="wide"
)

# Header Section
st.title(f"{ICON_PLANET} Regulus")
st.caption("Photometric light curve analysis and automated exoplanetary transit detection.")

# Sidebar Configuration & Information
with st.sidebar:
    st.header("About Regulus")
    st.markdown(
        """
        **Regulus** analyzes stellar brightness time-series (light curves) to identify exoplanet transits.
        
        **Pipeline Overview:**
        1. **Detrending**: Stellar variability filtering via `lightkurve`.
        2. **BLS Search**: Box Least Squares periodogram algorithm to detect periodic box-shaped dips.
        3. **ML Classifier**: Balanced Random Forest model scoring candidate signals.
        4. **Persistence**: Local SQLite logging of analysis sessions.
        """
    )
    st.divider()
    st.markdown("Developed with Python, Streamlit, Astropy, Lightkurve, and Scikit-Learn.")

tab_analyze, tab_history = st.tabs([f"{ICON_ANALYZE} Analyze", f"{ICON_HISTORY} History"])

# -------------------------------------------------------------
# TAB 1: ANALYZE
# -------------------------------------------------------------
with tab_analyze:
    st.subheader("Light Curve Ingestion")

    input_mode = st.radio(
        "Choose data source:",
        ["Upload CSV File", "Select Bundled Sample"],
        horizontal=True
    )

    df_data = None
    source_name = ""

    if input_mode == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "Upload a light curve CSV containing 'time' and 'flux' columns",
            type=["csv"]
        )
        if uploaded_file is not None:
            try:
                df_data = pd.read_csv(uploaded_file)
                source_name = uploaded_file.name
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
    else:
        sample_files = sorted(glob.glob(str(PROJECT_ROOT / "data" / "sample_light_curves" / "*.csv")))
        sample_options = {os.path.basename(f): f for f in sample_files}
        
        if sample_options:
            selected_sample = st.selectbox(
                "Choose a benchmark light curve:",
                list(sample_options.keys()),
                format_func=lambda x: {
                    "sample_kepler_transit.csv": f"{ICON_PLANET} Kepler Transit Candidate (Positive)",
                    "sample_quiet_star.csv": f"{ICON_STAR} Quiet Star Baseline (Negative)",
                    "sample_variable_star.csv": f"{ICON_SPARKLE} Variable / Pulsating Star (Negative)"
                }.get(x, x)
            )
            if selected_sample:
                sample_path = sample_options[selected_sample]
                df_data = pd.read_csv(sample_path)
                source_name = selected_sample
        else:
            st.warning("No sample files found in data/sample_light_curves/.")

    if df_data is not None:
        try:
            raw_time, raw_flux = parse_light_curve_frame(df_data)
        except ValueError as e:
            st.error(str(e))
            raw_time = None

        if raw_time is not None:
            t_span = np.nanmax(raw_time) - np.nanmin(raw_time)
            st.success(f"Loaded **{source_name}** ({len(df_data):,} observations, {t_span:.2f} days time-span).")

            with st.spinner("Executing detrending, Box Least Squares periodogram, and ML inference..."):
                try:
                    # 1. Preprocessing & Detrending
                    lc = load_light_curve(raw_time, raw_flux)

                    # 2. BLS Feature Extraction
                    features = extract_features(lc)

                    # 3. ML Model Inference
                    result = predict(features)

                    # 4. Save to Database
                    save_analysis(source_name, result)

                except Exception as e:
                    st.error(f"Pipeline execution failed: {e}")
                    result = None

            if result is not None:
                st.divider()
                st.subheader("Analysis Results")

                is_planet = (result["prediction"] == "exoplanet")
                conf_pct = result["confidence"] * 100

                # Verdict Badge
                if is_planet:
                    st.success(
                        f"### {ICON_PLANET} Verdict: Exoplanet Transit Candidate Detected\n"
                        f"**Model Confidence:** `{conf_pct:.1f}%` likelihood based on Box Least Squares transit profile."
                    )
                else:
                    st.info(
                        f"### {ICON_STAR} Verdict: No Exoplanet Transit Detected\n"
                        f"**Model Confidence:** `{conf_pct:.1f}%` (classified as stellar baseline or non-transit variability)."
                    )

                # Metric Cards
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Orbital Period", f"{result['period']:.3f} days")
                col2.metric("Transit Depth", f"{(result['depth'] * 100):.3f}%", help="Relative drop in stellar flux during transit.")
                col3.metric("Transit Duration", f"{(result['duration'] * 24):.2f} hours", f"{result['duration']:.3f} days")
                col4.metric("BLS Peak Power", f"{result['bls_power']:.4f}", help="Signal-to-noise strength of the detected periodicity.")

                # Interactive Visualization
                st.subheader("Interactive Light Curve")

                time_vals = lc.time.value
                flux_vals = lc.flux.value

                fig = go.Figure()

                # Flattened Light Curve scatter
                fig.add_trace(go.Scattergl(
                    x=time_vals,
                    y=flux_vals,
                    mode="markers",
                    marker=dict(size=3, color="#38bdf8", opacity=0.7),
                    name="Detrended Flux"
                ))

                # If transit detected, highlight transit epochs
                if is_planet and result["period"] > 0:
                    period_val = result["period"]
                    t0_val = result["transit_time"]
                    dur_val = result["duration"]
                    t_min, t_max = time_vals.min(), time_vals.max()

                    # Calculate transit centers within the time range
                    n_start = int(np.floor((t_min - t0_val) / period_val))
                    n_end = int(np.ceil((t_max - t0_val) / period_val))

                    for n in range(n_start, n_end + 1):
                        transit_center = t0_val + n * period_val
                        if t_min <= transit_center <= t_max:
                            fig.add_vrect(
                                x0=transit_center - (dur_val / 2.0),
                                x1=transit_center + (dur_val / 2.0),
                                fillcolor="#f59e0b",
                                opacity=0.25,
                                layer="below",
                                line_width=1,
                                line_color="#d97706"
                            )

                fig.update_layout(
                    title=f"Normalized Flux vs Time — {source_name}",
                    xaxis_title="Time (Days / BJD)",
                    yaxis_title="Normalized Relative Flux",
                    template="plotly_dark",
                    hovermode="x",
                    height=450,
                    margin=dict(l=40, r=40, t=50, b=40)
                )

                st.plotly_chart(fig, use_container_width=True)

                # Phase-folded transit view
                with st.expander(f"{ICON_SEARCH} View Phase-Folded Transit Profile"):
                    phase_period = result["period"]
                    t0_val = result["transit_time"]
                    folded_phase = (((time_vals - t0_val + 0.5 * phase_period) % phase_period) - 0.5 * phase_period) / phase_period

                    fig_fold = go.Figure()
                    fig_fold.add_trace(go.Scattergl(
                        x=folded_phase,
                        y=flux_vals,
                        mode="markers",
                        marker=dict(size=3, color="#a78bfa", opacity=0.7),
                        name="Phase-Folded Flux"
                    ))
                    fig_fold.update_layout(
                        title=f"Phase-Folded Light Curve (Period = {phase_period:.3f} days)",
                        xaxis_title="Orbital Phase (-0.5 to +0.5)",
                        yaxis_title="Normalized Relative Flux",
                        template="plotly_dark",
                        height=400,
                        margin=dict(l=40, r=40, t=50, b=40)
                    )
                    st.plotly_chart(fig_fold, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: HISTORY
# -------------------------------------------------------------
with tab_history:
    st.subheader("Analysis Run History")

    history_records = load_history()

    if not history_records:
        st.info("No analysis records found in the database. Run an analysis in the 'Analyze' tab to generate records.")
    else:
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        total_runs = len(history_records)
        pos_runs = sum(1 for r in history_records if r.get("prediction") == "exoplanet")
        neg_runs = total_runs - pos_runs

        col_stat1.metric("Total Analyses", total_runs)
        col_stat2.metric("Exoplanet Candidates", pos_runs)
        col_stat3.metric("Non-Transits", neg_runs)

        st.divider()

        df_hist = pd.DataFrame(history_records)
        
        # Format display columns
        df_display = pd.DataFrame({
            "ID": df_hist["id"],
            "Timestamp (UTC)": df_hist["created_at"],
            "Dataset / File": df_hist["filename"],
            "Verdict": df_hist["prediction"].apply(lambda p: f"{ICON_PLANET} Exoplanet" if p == "exoplanet" else f"{ICON_STAR} No Exoplanet"),
            "Confidence": (df_hist["confidence"] * 100).round(1).astype(str) + "%",
            "Period (days)": df_hist["period"].round(3),
            "Depth (%)": (df_hist["transit_depth"] * 100).round(3),
            "BLS Power": df_hist["bls_power"].round(4)
        })

        st.dataframe(df_display, use_container_width=True, hide_index=True)

        if st.button("Clear History Log", type="secondary"):
            clear_history()
            st.rerun()

# -------------------------------------------------------------
# FOOTER SECTION
# -------------------------------------------------------------
st.markdown("---")
col_footer_left, col_footer_right = st.columns([3, 2])
with col_footer_left:
    st.markdown(
        """
        <div style="font-size: 0.85rem; color: #94a3b8;">
            <strong>Developer:</strong> Piyush Sahu &nbsp;|&nbsp; <em>Project for Mini Project Lab</em>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_footer_right:
    st.markdown(
        """
        <div style="font-size: 0.85rem; color: #64748b; text-align: right;">
            Regulus Photometric Analysis System
        </div>
        """,
        unsafe_allow_html=True
    )
