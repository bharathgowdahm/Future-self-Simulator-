"""FutureSelf Simulator - Day 2 of 150-Day Challenge.
Monte Carlo scenario modeling with percentile bands.
Mathematical illustration only, not a real world prediction.
"""

import io
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Page config and theme ----------------
st.set_page_config(
    page_title="FutureSelf Simulator",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
   .main-header { font-size: 2.2rem; font-weight: 800; margin-bottom: 0; }
   .sub-header { color: #888; margin-top: 0; }
   .metric-card {
        background: var(--secondary-background-color, #f0f2f6);
        border-radius: 12px; padding: 16px;
        border: 1px solid rgba(128,128,128,.2);
    }
   .disclaimer {
        background: #fff8e1; color: #5d4037; border-radius: 8px;
        padding: 12px 16px; border-left: 4px solid #ffa000; font-size:.9rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown('<p class="main-header">🔮 FutureSelf Simulator</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Scenario modeling with Monte Carlo uncertainty analysis. '
    'Day 2 of the 150-Day Expert GitHub Challenge.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="disclaimer">⚠️ These are <b>mathematical simulations</b> of the assumptions '
    'you enter. They are illustrative scenarios, not predictions of your real future.</div>',
    unsafe_allow_html=True,
)

# ---------------- Seeded RNG helpers ----------------
def make_rng(seed: int | None):
    """Return a numpy Generator. Seeded for reproducibility, fresh entropy otherwise."""
    if seed is not None:
        return np.random.default_rng(seed)
    return np.random.default_rng()

def validate_inputs(start, mu, sigma, months, runs):
    errors = []
    if not 0 <= start <= 100:
        errors.append("Starting score must be between 0 and 100.")
    if not -20 <= mu <= 20:
        errors.append("Monthly improvement mean must be between -20 and +20.")
    if not 0 <= sigma <= 15:
        errors.append("Uncertainty (std-dev) must be between 0 and 15.")
    if not 1 <= months <= 120:
        errors.append("Duration must be between 1 and 120 months.")
    if not 100 <= runs <= 5000:
        errors.append("Simulation runs must be between 100 and 5,000.")
    return errors

# ---------------- Core simulation ----------------
def run_monte_carlo(start, mu, sigma, months, runs, seed, progress_cb=None):
    """Run Monte Carlo paths. Returns array of shape (runs, months+1)."""
    rng = make_rng(seed)
    paths = np.empty((runs, months + 1), dtype=np.float64)
    paths[:, 0] = start
    chunk = max(1, runs // 20)
    for i in range(runs):
        noise = rng.normal(loc=mu, scale=sigma, size=months)
        path = np.clip(start + np.cumsum(noise), 0, 100)
        paths[i, 1:] = path
        if progress_cb and (i + 1) % chunk == 0:
            progress_cb((i + 1) / runs)
    return paths

def percentile_bands(paths):
    months = np.arange(paths.shape[1])
    p10 = np.percentile(paths, 10, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p90 = np.percentile(paths, 90, axis=0)
    return pd.DataFrame({"month": months, "p10": p10, "p50": p50, "p90": p90})

def compute_probabilities(paths, target):
    final = paths[:, -1]
    p_final = float(np.mean(final >= target))
    p_ever = float(np.mean(np.any(paths >= target, axis=1)))
    return {
        "P(final >= target)": p_final,
        "P(ever reaches target)": p_ever,
        "median_final": float(np.median(final)),
        "mean_final": float(np.mean(final)),
        "best_final": float(np.max(final)),
        "worst_final": float(np.min(final)),
    }

# ---------------- Sidebar inputs ----------------
with st.sidebar:
    st.header("⚙️ Scenario Inputs")
    scenario_name = st.text_input("Scenario name", value="Learn guitar")
    start = st.slider("Starting score (0-100)", 0.0, 100.0, 20.0, 1.0)
    mu = st.slider("Monthly improvement (mean)", -5.0, 10.0, 2.0, 0.1)
    sigma = st.slider("Uncertainty (std-dev)", 0.0, 10.0, 1.5, 0.1)
    months = st.slider("Duration (months)", 1, 120, 24, 1)
    runs = st.select_slider(
        "Simulation runs", options=[100, 500, 1000, 2000, 5000], value=1000
    )
    use_seed = st.checkbox("Use random seed (reproducible)", value=True)
    seed = st.number_input("Seed", value=42, step=1) if use_seed else None
    target = st.slider("Target score", 0.0, 100.0, 80.0, 1.0)
    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

st.sidebar.divider()
st.sidebar.caption(
    "Math: score(t+1) = clip(score(t) + N(μ, σ), 0, 100). "
    "Percentiles computed per month across all runs."
)

# Session state for saved scenarios
if "saved" not in st.session_state:
    st.session_state.saved = {}

# ---------------- Run ----------------
if run_btn:
    errors = validate_inputs(start, mu, sigma, months, runs)
    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    config = dict(
        scenario=scenario_name, start=start, mu=mu, sigma=sigma,
        months=months, runs=runs, seed=seed, target=target,
    )
    progress = st.progress(0, text="Simulating paths...")
    t0 = time.time()
    paths = run_monte_carlo(start, mu, sigma, months, runs, seed,
                            progress_cb=lambda f: progress.progress(f, text=f"Simulating... {f:.0%}"))
    elapsed = time.time() - t0
    progress.empty()

    bands = percentile_bands(paths)
    probs = compute_probabilities(paths, target)

    st.session_state["last_result"] = {
        "config": config, "bands": bands, "probs": probs, "elapsed": elapsed,
    }
    st.session_state.saved[scenario_name] = {
        "config": config, "bands": bands, "probs": probs,
    }
    st.success(f"Done: {runs:,} runs × {months} months in {elapsed:.2f}s")

result = st.session_state.get("last_result")

if result is None:
    st.info("👈 Configure your scenario in the sidebar and press **Run Simulation**.")
    st.stop()

bands = result["bands"]
probs = result["probs"]
config = result["config"]

# ---------------- KPI row ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Median final score", f"{probs['median_final']:.1f}")
c2.metric("P(final ≥ target)", f"{probs['P(final >= target)']:.1%}")
c3.metric("P(ever reaches target)", f"{probs['P(ever reaches target)']:.1%}")
c4.metric("90th percentile final", f"{bands['p90'].iloc[-1]:.1f}")

# ---------------- Chart ----------------
st.subheader("📈 Percentile bands (P10 / P50 / P90)")
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.fill_between(bands["month"], bands["p10"], bands["p90"], alpha=0.25, label="P10–P90 band")
ax.plot(bands["month"], bands["p50"], linewidth=2.2, label="P50 (median)")
ax.plot(bands["month"], bands["p10"], linestyle="--", linewidth=1.2, label="P10")
ax.plot(bands["month"], bands["p90"], linestyle="--", linewidth=1.2, label="P90")
ax.axhline(target, color="red", linestyle=":", linewidth=1.5, label=f"Target {target:.0f}")
ax.set_xlabel("Month")
ax.set_ylabel("Score")
ax.set_ylim(0, 100)
ax.legend(loc="best")
ax.grid(alpha=0.3)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)

# ---------------- Tabs: details, sensitivity, compare, export ----------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Outcome details", "🎚️ Sensitivity", "⚖️ Compare scenarios", "💾 Export"]
)

with tab1:
    st.write("**Probability and outcome summary**")
    st.dataframe(pd.DataFrame([probs]).T.rename(columns={0: "value"}), use_container_width=True)
    st.write("**Percentile table (first 12 months)**")
    st.dataframe(bands.head(12), use_container_width=True, hide_index=True)

with tab2:
    st.write("How sensitive is the median final score to the monthly improvement mean?")
    deltas = st.slider("Sweep range (±)", 0.5, 3.0, 1.5, 0.5)
    mus = [mu - deltas, mu - deltas / 2, mu, mu + deltas / 2, mu + deltas]
    sens_rows = []
    rng_s = make_rng(seed)
    for m in mus:
        noise = rng_s.normal(loc=m, scale=sigma, size=(min(runs, 1000), months))
        finals = np.clip(start + np.cumsum(noise, axis=1)[:, -1], 0, 100)
        sens_rows.append({"monthly_mean": round(m, 2), "median_final": round(float(np.median(finals)), 1)})
    sens_df = pd.DataFrame(sens_rows)
    st.bar_chart(sens_df.set_index("monthly_mean"))
    st.dataframe(sens_df, use_container_width=True, hide_index=True)

with tab3:
    st.write("Saved scenarios overlay their P50 lines for comparison.")
    if len(st.session_state.saved) < 1:
        st.info("Run at least one simulation to compare.")
    else:
        fig2, ax2 = plt.subplots(figsize=(10, 4.2))
        summary = []
        for name, r in st.session_state.saved.items():
            b = r["bands"]
            ax2.plot(b["month"], b["p50"], linewidth=2, label=name)
            summary.append({
                "scenario": name,
                "median_final": round(r["probs"]["median_final"], 1),
                "P(final ≥ target)": round(r["probs"]["P(final >= target)"], 3),
            })
        ax2.set_xlabel("Month")
        ax2.set_ylabel("P50 score")
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
        if st.button("🗑 Clear saved scenarios"):
            st.session_state.saved = {}
            st.rerun()

with tab4:
    st.write("Download your results.")
    csv_buf = io.StringIO()
    bands.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇ Download CSV (month, p10, p50, p90)",
        data=csv_buf.getvalue(),
        file_name=f"{scenario_name.replace(' ', '_')}_bands.csv",
        mime="text/csv",
        use_container_width=True,
    )
    payload = {
        "config": {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in config.items()},
        "probabilities": probs,
        "bands": bands.to_dict(orient="records"),
        "note": "Mathematical simulation, not a real world prediction.",
    }
    st.download_button(
        "⬇ Download JSON (config + results)",
        data=json.dumps(payload, indent=2, default=float),
        file_name=f"{scenario_name.replace(' ', '_')}_results.json",
        mime="application/json",
        use_container_width=True,
    )

st.divider()
st.caption(
    "FutureSelf Simulator · Day 2 · Built with Streamlit, NumPy, Pandas, Matplotlib. "
    "Seeded RNG (NumPy PCG64) makes runs reproducible when a seed is set."
)
