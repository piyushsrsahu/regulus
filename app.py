"""
Regulus web application.
"""
import glob
import hashlib
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import PROJECT_ROOT
from src.preprocessing import load_light_curve, parse_light_curve_frame
from src.features import extract_features, estimate_planet_size
from src.model import predict
from src.db import init_db, save_analysis, load_history, clear_history

# Unicode character codes for interface symbols
ICON_PLANET = "\U0001FA90"     # Ringed Planet
ICON_STAR = "⭐"           # Star
ICON_SPARKLE = "✨"        # Sparkles
ICON_ANALYZE = "\U0001F4CA"    # Chart
ICON_HISTORY = "\U0001F4DC"    # History Scroll
ICON_SEARCH = "\U0001F50D"     # Magnifying Glass
ICON_TELESCOPE = "\U0001F52D"  # Telescope

# Defaults match the settings the classifier was trained with (train_model.py).
DEFAULT_WINDOW = 101
DEFAULT_MIN_PERIOD = 0.5

TRANSIT_FILL = "#f59e0b"
TRANSIT_LINE = "#d97706"
COLOR_RAW = "#94a3b8"
COLOR_DETRENDED = "#38bdf8"
COLOR_BLS = "#a855f7"
COLOR_CANDIDATE = "#22c55e"
COLOR_NEGATIVE = "#64748b"

SAMPLE_LABELS = {
    "sample_kepler_transit.csv": f"{ICON_PLANET} Synthetic Transit, Kepler-like cadence (Positive)",
    "sample_quiet_star.csv": f"{ICON_STAR} Quiet Star Baseline (Negative)",
    "sample_variable_star.csv": f"{ICON_SPARKLE} Variable / Pulsating Star (Negative)",
}


@st.cache_data(show_spinner=False)
def run_pipeline(time: np.ndarray, flux: np.ndarray, window_length: int,
                 min_period: float, max_period: float | None) -> dict:
    """Detrend, run BLS and classify. Cached so widget reruns don't recompute."""
    valid = np.isfinite(time) & np.isfinite(flux)
    order = np.argsort(time[valid])
    raw_t, raw_f = time[valid][order], flux[valid][order]

    lc = load_light_curve(time, flux, window_length=window_length)
    features, (pg_period, pg_power) = extract_features(
        lc, min_period=min_period, max_period=max_period, return_periodogram=True
    )
    result = predict(features)
    return {
        "raw_time": raw_t, "raw_flux": raw_f,
        "time": lc.time.value, "flux": lc.flux.value,
        "pg_period": pg_period, "pg_power": pg_power,
        "result": result,
    }


def transit_centers(t0: float, period: float, t_min: float, t_max: float) -> list[float]:
    if period <= 0:
        return []
    n_start = int(np.floor((t_min - t0) / period))
    n_end = int(np.ceil((t_max - t0) / period))
    return [t0 + n * period for n in range(n_start, n_end + 1) if t_min <= t0 + n * period <= t_max]


def peak_preserving_downsample(x: np.ndarray, y: np.ndarray, n_bins: int = 3000):
    """Keep the max of each chunk so narrow BLS peaks survive plotting."""
    if len(x) <= n_bins:
        return x, y
    chunks = np.array_split(np.arange(len(x)), n_bins)
    idx = np.array([c[np.argmax(y[c])] for c in chunks])
    return x[idx], y[idx]


def binned_mean(x: np.ndarray, y: np.ndarray, n_bins: int = 60):
    bins = np.linspace(-0.5, 0.5, n_bins + 1)
    which = np.digitize(x, bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])
    means = np.array([y[which == i].mean() if np.any(which == i) else np.nan for i in range(n_bins)])
    keep = np.isfinite(means)
    return centers[keep], means[keep]


# Initialize local SQLite storage
init_db()

st.set_page_config(page_title="Regulus", page_icon=ICON_PLANET, layout="wide")

st.title(f"{ICON_PLANET} Regulus")
st.caption("Photometric light curve analysis and automated exoplanetary transit detection.")

# -------------------------------------------------------------
# SIDEBAR: about + analysis parameters
# -------------------------------------------------------------
with st.sidebar:
    st.header("About Regulus")
    st.markdown(
        """
        **Regulus** detects exoplanets by discovering periodic dips in starlight caused by an
        orbiting planet transiting its host star.

        **Pipeline Overview:**
        1. **Detrending**: Savitzky-Golay flattening via `lightkurve`.
        2. **BLS Search**: Box Least Squares periodogram for periodic box-shaped dips.
        3. **ML Classifier**: Balanced Random Forest scoring the candidate signal.
        4. **Physical Modeling**: Orbital period, transit depth, duration and planetary radius.
        5. **Persistence**: Local SQLite log of every analysis.
        """
    )
    st.divider()
    st.subheader(f"{ICON_TELESCOPE} Analysis Parameters")
    param_window = st.slider(
        "Detrending Window Length", min_value=21, max_value=401, value=DEFAULT_WINDOW, step=2,
        help="Window size (points, odd) for Savitzky-Golay polynomial flattening.",
    )
    param_min_p = st.number_input(
        "Min Search Period (days)", min_value=0.1, max_value=50.0, value=DEFAULT_MIN_PERIOD, step=0.1,
        help="Shortest orbital period tested by the BLS search.",
    )
    param_max_p = st.number_input(
        "Max Search Period (days)", min_value=0.0, max_value=500.0, value=0.0, step=0.5,
        help="Longest orbital period tested. 0 = automatic (half the observation baseline).",
    )
    st.caption(
        "Defaults match the settings the classifier was trained with; "
        "changing them shifts the extracted features and can change the verdict."
    )
    st.divider()
    st.markdown("Developed with Python, Streamlit, Astropy, Lightkurve, and Scikit-Learn.")

tab_analyze, tab_history = st.tabs([f"{ICON_ANALYZE} Analyze", f"{ICON_HISTORY} History"])

# -------------------------------------------------------------
# TAB 1: ANALYZE
# -------------------------------------------------------------
with tab_analyze:
    st.subheader("1. Light Curve Ingestion")

    input_mode = st.radio("Choose data source:", ["Upload CSV File", "Select Bundled Sample"], horizontal=True)

    df_data = None
    source_name = ""

    if input_mode == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "Upload a light curve CSV containing 'time' and 'flux' columns", type=["csv"]
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
                format_func=lambda x: SAMPLE_LABELS.get(x, x),
            )
            if selected_sample:
                df_data = pd.read_csv(sample_options[selected_sample])
                source_name = selected_sample
        else:
            st.warning("No sample files found in data/sample_light_curves/.")

    out = None
    if df_data is not None:
        try:
            raw_time, raw_flux = parse_light_curve_frame(df_data)
        except ValueError as e:
            st.error(str(e))
            raw_time = None

        if raw_time is not None:
            t_span = np.nanmax(raw_time) - np.nanmin(raw_time)
            st.success(f"Loaded **{source_name}** ({len(df_data):,} observations, {t_span:.2f} days time-span).")

            with st.spinner("Processing photometric time-series, detrending stellar variability, "
                            "and running Box Least Squares spectrum..."):
                try:
                    out = run_pipeline(raw_time, raw_flux, int(param_window), float(param_min_p),
                                       float(param_max_p) if param_max_p > 0 else None)
                except Exception as e:
                    st.error(f"Pipeline execution failed: {e}")

            if out is not None:
                # Persist once per distinct input + parameters (not on every widget rerun).
                digest = hashlib.sha1(np.ascontiguousarray(raw_flux).tobytes()).hexdigest()
                run_key = (source_name, digest, int(param_window), float(param_min_p), float(param_max_p))
                if st.session_state.get("last_saved_run") != run_key:
                    save_analysis(source_name, out["result"])
                    st.session_state["last_saved_run"] = run_key

    if out is not None:
        result = out["result"]
        is_planet = result["prediction"] == "exoplanet"
        conf_pct = result["confidence"] * 100
        time_vals, flux_vals = out["time"], out["flux"]
        t_min, t_max = float(time_vals.min()), float(time_vals.max())
        centers = transit_centers(result["transit_time"], result["period"], t_min, t_max) if is_planet else []

        # ---------------- 2. Verdict & confidence ----------------
        st.divider()
        st.subheader("2. Detection Verdict & Confidence")
        col_verdict, col_gauge = st.columns([3, 2])

        with col_verdict:
            if is_planet:
                st.success(
                    f"### {ICON_PLANET} Exoplanet Transit Candidate Detected\n"
                    "The algorithmic pipeline recovered a periodic U-shaped transit signature "
                    "consistent with an orbiting planetary body."
                )
                size = estimate_planet_size(result["depth"])
                st.markdown(
                    f"""
                    <div style="background-color: rgba(30, 41, 59, 0.85); padding: 15px; border-radius: 8px;
                                border: 1px solid #334155; margin-top: 10px; color: #e2e8f0;">
                        <div style="font-weight: 600; color: #cbd5e1; margin-bottom: 6px;">
                            Estimated Physical Properties
                            <span style="font-weight: 400;">(Assuming Solar-Type Host Star):</span>
                        </div>
                        <span style="color: #38bdf8; font-weight: 700; font-size: 1.15rem;">{size['radius_earth']:.2f} R⊕</span>
                        <span style="font-size: 0.8rem; color: #cbd5e1;">(Earth Radii)</span>&nbsp;&nbsp;&nbsp;
                        <span style="color: #f59e0b; font-weight: 700; font-size: 1.15rem;">{size['radius_jupiter']:.2f} RJ</span>
                        <span style="font-size: 0.8rem; color: #cbd5e1;">(Jupiter Radii)</span>&nbsp;&nbsp;&nbsp;
                        <span style="color: #c084fc; font-weight: 700;">Class:</span> {size['planet_class']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info(
                    f"### {ICON_STAR} No Exoplanet Transit Detected\n"
                    "The signal is consistent with a stellar baseline or non-transit variability; "
                    "no periodic transit-shaped dip was strong enough to classify as a candidate."
                )

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf_pct,
                number={"suffix": "%", "valueformat": ".1f"},
                title={"text": "Classification Confidence"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": COLOR_CANDIDATE if is_planet else COLOR_NEGATIVE},
                    "steps": [{"range": [0, 50], "color": "rgba(148,163,184,0.25)"},
                              {"range": [50, 100], "color": "rgba(148,163,184,0.12)"}],
                },
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=30, r=30, t=60, b=10))
            st.plotly_chart(fig_gauge, width="stretch")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Orbital Period (P)", f"{result['period']:.3f} days", help="Period at the BLS power maximum.")
        m2.metric("Transit Dip Depth (δ)", f"{result['depth'] * 100:.3f}%", help="Relative drop in stellar flux during transit.")
        m3.metric("Transit Duration (T)", f"{result['duration'] * 24:.2f} hrs", f"{result['duration']:.3f} days",
                  delta_color="off", help="Best-fit box duration.")
        m4.metric("BLS Spectral Power", f"{result['bls_power']:.4f}", help="Signal peak strength above background noise.")

        # ---------------- 3. Diagnostics ----------------
        st.divider()
        st.subheader("3. Comprehensive Visual Diagnostics")

        st.markdown("#### Comparison: Raw Signal vs Detrended Flux")
        st.caption("Visual proof of stellar detrending: top panel shows raw observations; bottom panel shows "
                   "flattened flux with detected transit windows highlighted in amber.")
        fig_dual = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
            subplot_titles=("Raw Photometric Observations (with Stellar Drift / Trends)",
                            "Detrended & Normalized Flux (Flattened Baseline ~ 1.0)"),
        )
        fig_dual.add_trace(go.Scattergl(x=out["raw_time"], y=out["raw_flux"], mode="markers",
                                        marker=dict(size=3, color=COLOR_RAW, opacity=0.8), name="Raw Flux"), row=1, col=1)
        fig_dual.add_trace(go.Scattergl(x=time_vals, y=flux_vals, mode="markers",
                                        marker=dict(size=3, color=COLOR_DETRENDED, opacity=0.8), name="Detrended Flux"),
                           row=2, col=1)
        for tc in centers:
            fig_dual.add_vrect(x0=tc - result["duration"] / 2, x1=tc + result["duration"] / 2,
                               fillcolor=TRANSIT_FILL, opacity=0.25, layer="below", line_width=1,
                               line_color=TRANSIT_LINE, row=2, col=1)
        fig_dual.update_yaxes(title_text="Observed Flux", row=1, col=1)
        fig_dual.update_yaxes(title_text="Normalized Flux", row=2, col=1)
        fig_dual.update_xaxes(title_text="Time (Days / BJD)", row=2, col=1)
        fig_dual.update_layout(height=620, hovermode="x", margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(fig_dual, width="stretch")

        col_chart_left, col_chart_right = st.columns(2)

        with col_chart_left:
            st.markdown("#### Phase-Folded Transit Profile")
            st.caption("All observed transit events folded onto a single orbital cycle (-0.5 to +0.5 phase) "
                       "with binned mean curve.")
            P, t0 = result["period"], result["transit_time"]
            folded_phase = (((time_vals - t0 + 0.5 * P) % P) - 0.5 * P) / P
            bin_centers, binned_flux = binned_mean(folded_phase, flux_vals)
            fig_phase = go.Figure()
            fig_phase.add_trace(go.Scattergl(x=folded_phase, y=flux_vals, mode="markers",
                                             marker=dict(size=3, color=COLOR_RAW, opacity=0.45), name="All Observations"))
            fig_phase.add_trace(go.Scatter(x=bin_centers, y=binned_flux, mode="lines+markers",
                                           line=dict(color=TRANSIT_FILL, width=3), marker=dict(size=5, color=TRANSIT_LINE),
                                           name="Binned Average"))
            fig_phase.update_layout(title=f"Phase-Folded Curve (Period = {P:.3f} d)",
                                    xaxis_title="Orbital Phase (-0.5 to +0.5)", yaxis_title="Normalized Flux",
                                    height=430, margin=dict(l=40, r=20, t=50, b=40))
            st.plotly_chart(fig_phase, width="stretch")

        with col_chart_right:
            st.markdown("#### BLS Periodogram Power Spectrum")
            st.caption("Box Least Squares spectral power across test periods. "
                       "The prominent peak marks the detected orbital period.")
            order = np.argsort(out["pg_period"])
            px_, py_ = peak_preserving_downsample(out["pg_period"][order], out["pg_power"][order])
            fig_bls = go.Figure(go.Scatter(x=px_, y=py_, mode="lines", line=dict(color=COLOR_BLS, width=2),
                                           name="BLS Power"))
            fig_bls.add_vline(x=result["period"], line_dash="dash", line_color=COLOR_DETRENDED,
                              annotation_text=f"P = {result['period']:.2f} d", annotation_position="top right")
            fig_bls.update_layout(title=f"BLS Power Spectrum (Peak Power = {result['bls_power']:.4f})",
                                  xaxis_title="Test Period (Days)", yaxis_title="BLS Power",
                                  height=430, margin=dict(l=40, r=20, t=50, b=40), showlegend=False)
            st.plotly_chart(fig_bls, width="stretch")

        with st.expander(f"{ICON_SEARCH} Close-Up of the Primary Transit Window", expanded=is_planet):
            tc = centers[0] if centers else result["transit_time"]
            zoom_window = max(3.0 * result["duration"], 0.3)
            mask_zoom = np.abs(time_vals - tc) <= zoom_window
            if not np.any(mask_zoom):
                st.info("No data points fall within the primary transit center window.")
            else:
                fig_zoom = go.Figure(go.Scatter(x=time_vals[mask_zoom], y=flux_vals[mask_zoom], mode="markers",
                                                marker=dict(size=6, color=COLOR_DETRENDED), name="Detrended Flux"))
                fig_zoom.add_vrect(x0=tc - result["duration"] / 2, x1=tc + result["duration"] / 2,
                                   fillcolor=TRANSIT_FILL, opacity=0.2, layer="below", line_width=0)
                fig_zoom.add_hline(y=1.0, line_dash="dot", line_color=COLOR_RAW, annotation_text="Baseline")
                fig_zoom.add_hline(y=1.0 - result["depth"], line_dash="dot", line_color=TRANSIT_LINE,
                                   annotation_text=f"Depth {result['depth'] * 100:.2f}%", annotation_position="bottom right")
                fig_zoom.update_layout(title=f"Close-Up Transit Window at t₀ = {tc:.3f} d",
                                       xaxis_title="Time (Days / BJD)", yaxis_title="Normalized Flux",
                                       height=380, margin=dict(l=40, r=40, t=50, b=40))
                st.plotly_chart(fig_zoom, width="stretch")

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
        df_display = pd.DataFrame({
            "ID": df_hist["id"],
            "Timestamp (UTC)": df_hist["created_at"],
            "Dataset / File": df_hist["filename"],
            "Verdict": df_hist["prediction"].apply(
                lambda p: f"{ICON_PLANET} Exoplanet" if p == "exoplanet" else f"{ICON_STAR} No Exoplanet"),
            "Confidence": (df_hist["confidence"] * 100).round(1).astype(str) + "%",
            "Period (days)": df_hist["period"].round(3),
            "Depth (%)": (df_hist["transit_depth"] * 100).round(3),
            "BLS Power": df_hist["bls_power"].round(4),
        })

        st.dataframe(df_display, width="stretch", hide_index=True)

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
        unsafe_allow_html=True,
    )
with col_footer_right:
    st.markdown(
        """
        <div style="font-size: 0.85rem; color: #64748b; text-align: right;">
            Regulus Photometric Analysis System
        </div>
        """,
        unsafe_allow_html=True,
    )
