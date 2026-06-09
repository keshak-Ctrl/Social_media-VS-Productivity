"""
╔══════════════════════════════════════════════════════════════╗
║   SOCIAL MEDIA VS PRODUCTIVITY  |  STREAMLIT DASHBOARD      ║
╠══════════════════════════════════════════════════════════════╣
║   Run:  streamlit run app.py                                 ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from scipy.interpolate import make_interp_spline
import joblib, os, json, warnings

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "Social Media vs Productivity",
    page_icon   = "📊",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Colour tokens ─────────────────────────────────────────────
C = dict(
    primary="#1A6B8A", accent="#F4A261", danger="#E63946",
    success="#2A9D8F", purple="#534AB7", dark="#1A1A2E",
    muted="#6C757D",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

  .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }

  /* Hero banner */
  .hero-banner {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    border: 1px solid rgba(26,107,138,0.4);
  }
  .hero-title {
    font-size: 2.2rem; font-weight: 800; letter-spacing: -0.03em;
    background: linear-gradient(90deg, #F8F9FA, #74C0FC);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
  }
  .hero-sub {
    color: rgba(255,255,255,0.65); font-size: 1rem; margin-top: 0.4rem;
  }

  /* KPI cards */
  .kpi-card {
    background: white; border-radius: 12px; padding: 1.2rem 1.4rem;
    border-left: 4px solid; box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    height: 100%;
  }
  .kpi-value { font-size: 2rem; font-weight: 800; line-height: 1; }
  .kpi-label { font-size: 0.78rem; color: #6C757D; font-weight: 600;
               text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.3rem; }
  .kpi-delta { font-size: 0.85rem; margin-top: 0.4rem; }

  /* Section headers */
  .section-header {
    font-size: 1.15rem; font-weight: 700; color: #1A1A2E;
    border-bottom: 2px solid #F4A261; padding-bottom: 0.35rem;
    margin: 1.4rem 0 0.9rem;
  }

  /* Prediction box */
  .pred-box {
    border-radius: 12px; padding: 1.4rem; text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
  }
  .pred-score { font-size: 3.5rem; font-weight: 800; line-height: 1; }
  .pred-tier  { font-size: 1.1rem; font-weight: 700; margin-top: 0.4rem; }

  /* Insight callout */
  .insight {
    background: #F0FAF8; border-left: 4px solid #2A9D8F;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin: 0.6rem 0;
    font-size: 0.88rem; color: #2C3E50;
  }
  .insight strong { color: #2A9D8F; }

  /* Warning callout */
  .warn {
    background: #FFF8F0; border-left: 4px solid #F4A261;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin: 0.6rem 0;
    font-size: 0.88rem; color: #5C3D11;
  }

  /* Sidebar */
  [data-testid="stSidebar"] { background: #F0F4F8; }
  .sidebar-title { font-weight: 800; font-size: 1.2rem; color: #1A1A2E; }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0; padding: 10px 20px;
    font-weight: 600; font-size: 0.88rem;
  }

  /* Metric override */
  [data-testid="stMetricValue"] { font-family: 'Plus Jakarta Sans', sans-serif; }

  code, .stCode { font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    path = "data/final_with_personas.csv"
    if not os.path.exists(path):
        path = "data/engineered_social_media.csv"
    if not os.path.exists(path):
        path = "data/cleaned_social_media.csv"
    df = pd.read_csv(path)

    # Ensure persona column
    if "persona_cluster" not in df.columns:
        df["persona_cluster"] = 0

    # Standardise categoricals
    for col in ["gender", "job_type", "social_platform_preference"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Ensure productivity_tier
    if "productivity_tier" not in df.columns:
        df["productivity_tier"] = pd.cut(
            df["actual_productivity_score"], bins=[0, 4, 7, 10],
            labels=["Low", "Medium", "High"]
        )

    # Age group
    if "age_group" not in df.columns:
        df["age_group"] = pd.cut(
            df["age"], bins=[18, 25, 35, 45, 55, 100],
            labels=["18-25", "26-35", "36-45", "46-55", "56+"]
        )

    return df


@st.cache_resource
def load_models():
    models = {}
    for name, path in [
        ("regressor",  "models/productivity_regressor.pkl"),
        ("classifier", "models/productivity_classifier.pkl"),
        ("imputer",    "models/imputer.pkl"),
    ]:
        if os.path.exists(path):
            models[name] = joblib.load(path)
    if os.path.exists("models/feature_cols.pkl"):
        models["feature_cols"] = joblib.load("models/feature_cols.pkl")
    if os.path.exists("models/metrics.json"):
        with open("models/metrics.json") as f:
            models["metrics"] = json.load(f)
    return models


# ── plt helper ────────────────────────────────────────────────
def _despine(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#F8F9FA")


def _fig():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.facecolor": "#F8F9FA",
        "figure.facecolor": "white",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
    })


# ── Load everything ───────────────────────────────────────────
df     = load_data()
models = load_models()

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="sidebar-title">📊 Dashboard Controls</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown("**🔍 Filter Dataset**")

    # Gender filter
    genders = sorted(df["gender"].dropna().unique()) if "gender" in df.columns else ["All"]
    gender_sel = st.multiselect("Gender", genders, default=genders)

    # Job type filter
    if "job_type" in df.columns:
        jobs = sorted(df["job_type"].dropna().unique())
        job_sel = st.multiselect("Job Type", jobs, default=jobs)
    else:
        job_sel = []

    # Platform filter
    if "social_platform_preference" in df.columns:
        platforms = sorted(df["social_platform_preference"].dropna().unique())
        platform_sel = st.multiselect("Platform", platforms, default=platforms)
    else:
        platform_sel = []

    # Age range
    age_min = int(df["age"].min()) if "age" in df.columns else 18
    age_max = int(df["age"].max()) if "age" in df.columns else 65
    age_range = st.slider("Age Range", age_min, age_max, (age_min, age_max))

    st.divider()
    st.markdown("**📈 Chart Settings**")
    show_trend  = st.toggle("Show Trend Lines", value=True)
    show_ci     = st.toggle("Show Confidence Intervals", value=True)
    chart_theme = st.selectbox("Colour Theme", ["Default", "Muted", "Vibrant"])

    st.divider()
    st.markdown("**ℹ️ About**")
    st.caption("Social Media vs Productivity Analysis\nMSc Data Science Project")

# ── Apply filters ─────────────────────────────────────────────
fdf = df.copy()
if gender_sel and "gender" in fdf.columns:
    fdf = fdf[fdf["gender"].isin(gender_sel)]
if job_sel and "job_type" in fdf.columns:
    fdf = fdf[fdf["job_type"].isin(job_sel)]
if platform_sel and "social_platform_preference" in fdf.columns:
    fdf = fdf[fdf["social_platform_preference"].isin(platform_sel)]
if "age" in fdf.columns:
    fdf = fdf[fdf["age"].between(age_range[0], age_range[1])]

# ══════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
  <h1 class="hero-title">📱 Social Media vs Productivity</h1>
  <p class="hero-sub">
    A complete data science analysis · {n:,} participants · 7 novel engineered features · Random Forest ML models
  </p>
</div>
""".format(n=len(fdf)), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════
metrics = models.get("metrics", {})

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    avg_prod = fdf["actual_productivity_score"].mean()
    st.markdown(f"""
    <div class="kpi-card" style="border-color:{C['primary']}">
      <div class="kpi-value" style="color:{C['primary']}">{avg_prod:.2f}</div>
      <div class="kpi-label">Avg Productivity</div>
      <div class="kpi-delta">⚖️ out of 10</div>
    </div>""", unsafe_allow_html=True)

with k2:
    avg_usage = fdf["daily_social_media_time"].mean()
    usage_color = C["danger"] if avg_usage > 4 else C["accent"] if avg_usage > 2 else C["success"]
    st.markdown(f"""
    <div class="kpi-card" style="border-color:{usage_color}">
      <div class="kpi-value" style="color:{usage_color}">{avg_usage:.1f}h</div>
      <div class="kpi-label">Avg Daily SM Usage</div>
      <div class="kpi-delta">{'🔴 High risk' if avg_usage > 4 else '🟡 Moderate' if avg_usage > 2 else '🟢 Healthy'}</div>
    </div>""", unsafe_allow_html=True)

with k3:
    doomscroll_pct = (fdf["doomscroll_risk"].mean() * 100) if "doomscroll_risk" in fdf.columns else 0
    st.markdown(f"""
    <div class="kpi-card" style="border-color:{C['danger']}">
      <div class="kpi-value" style="color:{C['danger']}">{doomscroll_pct:.1f}%</div>
      <div class="kpi-label">Doomscroll Risk</div>
      <div class="kpi-delta">🌀 High-risk users</div>
    </div>""", unsafe_allow_html=True)

with k4:
    rf_r2 = metrics.get("rf_r2", None)
    r2_display = f"{rf_r2:.3f}" if rf_r2 else "—"
    st.markdown(f"""
    <div class="kpi-card" style="border-color:{C['success']}">
      <div class="kpi-value" style="color:{C['success']}">{r2_display}</div>
      <div class="kpi-label">Model R² Score</div>
      <div class="kpi-delta">🎯 RF Regression</div>
    </div>""", unsafe_allow_html=True)

with k5:
    clf_acc = metrics.get("clf_accuracy", None)
    acc_display = f"{clf_acc*100:.1f}%" if clf_acc else "—"
    st.markdown(f"""
    <div class="kpi-card" style="border-color:{C['purple']}">
      <div class="kpi-value" style="color:{C['purple']}">{acc_display}</div>
      <div class="kpi-label">Classifier Accuracy</div>
      <div class="kpi-delta">🏷️ Tier prediction</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview",
    "🔍 Deep Dive",
    "🧬 Novel Features",
    "🤖 ML Models",
    "🎯 Predict",
    "📖 Story",
])


# ═══════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Usage vs Productivity — The Core Relationship</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])

    with col_a:
        _fig()
        fig, ax = plt.subplots(figsize=(10, 5))
        x = fdf["daily_social_media_time"]; y = fdf["actual_productivity_score"]
        ax.scatter(x, y, alpha=0.20, s=14, color=C["primary"], zorder=2)

        if show_trend:
            bins = np.linspace(x.min(), x.max(), 22)
            bx   = (bins[:-1] + bins[1:]) / 2
            by   = [y[(x >= lo) & (x < hi)].mean() for lo, hi in zip(bins[:-1], bins[1:])]
            be   = [y[(x >= lo) & (x < hi)].sem()  for lo, hi in zip(bins[:-1], bins[1:])]
            valid = [not np.isnan(m) for m in by]
            bx2, by2, be2 = np.array(bx)[valid], np.array(by)[valid], np.array(be)[valid]
            try:
                spl = make_interp_spline(bx2, by2, k=3)
                xs  = np.linspace(bx2.min(), bx2.max(), 300)
                ax.plot(xs, spl(xs), color=C["danger"], lw=2.5, zorder=4, label="Smoothed trend")
                if show_ci:
                    ax.fill_between(bx2, by2 - be2, by2 + be2, alpha=0.15, color=C["danger"])
            except Exception:
                ax.plot(bx2, by2, color=C["danger"], lw=2.5)

        ax.axvspan(0, 2,        alpha=0.06, color=C["success"], label="Safe zone (<2h)")
        ax.axvspan(2, 4,        alpha=0.06, color=C["accent"],  label="Caution (2–4h)")
        ax.axvspan(4, x.max(),  alpha=0.08, color=C["danger"],  label="Risk zone (>4h)")
        r, p = stats.pearsonr(x, y)
        ax.set_xlabel("Daily Social Media Usage (hours)", fontsize=10)
        ax.set_ylabel("Actual Productivity Score", fontsize=10)
        ax.set_title(f"Usage vs Productivity  (Pearson r = {r:.3f},  p = {p:.4f})",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        _despine(ax); plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with col_b:
        r_val, p_val = stats.pearsonr(
            fdf["daily_social_media_time"], fdf["actual_productivity_score"]
        )
        effect = "strong" if abs(r_val) > 0.3 else "moderate" if abs(r_val) > 0.1 else "weak"
        direction = "negative" if r_val < 0 else "positive"

        st.markdown(f"""
        <div class="insight">
          <strong>Pearson r = {r_val:.3f}</strong><br>
          A {effect} {direction} correlation between social media usage and productivity.
        </div>
        <div class="insight">
          <strong>Tipping Point ~4h/day</strong><br>
          Productivity drops sharply beyond 4 hours of daily usage.
        </div>
        <div class="warn">
          <strong>Short-form platforms</strong> (TikTok, Instagram Reels) show the strongest
          negative association with productivity.
        </div>
        """, unsafe_allow_html=True)

        # Usage band breakdown
        fdf["usage_band"] = pd.cut(
            fdf["daily_social_media_time"],
            bins=[0, 1, 2, 4, 6, 24],
            labels=["<1h", "1-2h", "2-4h", "4-6h", ">6h"]
        )
        band_data = fdf.groupby("usage_band", observed=True)["actual_productivity_score"].mean()
        fig2, ax2 = plt.subplots(figsize=(5, 3.2))
        colors = [C["success"], C["success"], C["accent"], C["danger"], C["danger"]]
        bars = ax2.bar(band_data.index, band_data.values, color=colors, edgecolor="white", alpha=0.9)
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                     f"{bar.get_height():.2f}", ha="center", fontsize=8)
        ax2.set_title("Mean Productivity by Usage Band", fontsize=10)
        ax2.set_ylim(0, 10); _despine(ax2); plt.tight_layout()
        st.pyplot(fig2, use_container_width=True); plt.close()

    # ── Row 2: platform + demographics ────────────────────────
    st.markdown('<div class="section-header">Platform & Demographics Breakdown</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        if "social_platform_preference" in fdf.columns:
            platform_order = (
                fdf.groupby("social_platform_preference")["actual_productivity_score"]
                .median().sort_values(ascending=False).index
            )
            fig3, ax3 = plt.subplots(figsize=(7, 4))
            sns.boxplot(data=fdf, x="social_platform_preference", y="actual_productivity_score",
                        order=platform_order, palette="muted", ax=ax3)
            ax3.set_title("Productivity Distribution by Platform", fontsize=11)
            ax3.set_xlabel("Platform"); ax3.set_ylabel("Productivity Score")
            ax3.tick_params(axis="x", rotation=25); _despine(ax3); plt.tight_layout()
            st.pyplot(fig3, use_container_width=True); plt.close()

    with c2:
        if "age_group" in fdf.columns and "gender" in fdf.columns:
            fig4, ax4 = plt.subplots(figsize=(7, 4))
            sns.barplot(data=fdf, x="age_group", y="actual_productivity_score",
                        hue="gender", palette="Set2", errorbar="se", ax=ax4)
            ax4.set_title("Productivity by Age Group & Gender", fontsize=11)
            ax4.set_xlabel("Age Group"); ax4.set_ylabel("Mean Productivity Score")
            ax4.legend(title="Gender", fontsize=9)
            _despine(ax4); plt.tight_layout()
            st.pyplot(fig4, use_container_width=True); plt.close()


# ═══════════════════════════════════════════════════
# TAB 2 — DEEP DIVE
# ═══════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Lifestyle Factors & Correlation Analysis</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    pairs = [
        ("sleep_hours",          "Sleep Hours",          col1),
        ("stress_level",         "Stress Level",         col2),
        ("job_satisfaction_score","Job Satisfaction",    col3),
    ]
    for x_col, title, col in pairs:
        if x_col not in fdf.columns:
            continue
        with col:
            fig, ax = plt.subplots(figsize=(4.5, 4))
            scatter_kws = {"alpha": 0.2, "s": 14, "color": C["primary"]}
            line_kws    = {"color": C["danger"], "lw": 2}
            sns.regplot(data=fdf, x=x_col, y="actual_productivity_score",
                        scatter_kws=scatter_kws, line_kws=line_kws, ax=ax)
            r, _ = stats.pearsonr(fdf[x_col].dropna(), fdf.loc[fdf[x_col].notna(), "actual_productivity_score"])
            ax.set_title(f"{title}  (r={r:.3f})", fontsize=10)
            ax.set_ylabel("Productivity"); ax.set_xlabel(title)
            _despine(ax); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

    # Correlation heatmap
    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)

    HEATMAP_COLS = [c for c in [
        "actual_productivity_score", "daily_social_media_time",
        "sleep_hours", "stress_level", "number_of_notifications",
        "work_hours_per_day", "job_satisfaction_score",
        "coffee_consumption_per_day", "screen_time_before_sleep",
    ] if c in fdf.columns]

    corr = fdf[HEATMAP_COLS].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.5, annot_kws={"size": 9}, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    # Notifications deep dive
    st.markdown('<div class="section-header">Notification Interruption Effect</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.regplot(data=fdf, x="number_of_notifications", y="actual_productivity_score",
                    scatter_kws={"alpha": 0.2, "s": 14, "color": C["primary"]},
                    line_kws={"color": C["danger"], "lw": 2}, ax=ax)
        r, p = stats.pearsonr(fdf["number_of_notifications"], fdf["actual_productivity_score"])
        ax.set_title(f"Notifications vs Productivity  (r={r:.3f})", fontsize=11)
        ax.set_xlabel("Daily Notifications"); ax.set_ylabel("Productivity Score")
        _despine(ax); plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with c2:
        if "screen_time_before_sleep" in fdf.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.regplot(data=fdf, x="screen_time_before_sleep", y="sleep_hours",
                        scatter_kws={"alpha": 0.2, "s": 14, "color": C["purple"]},
                        line_kws={"color": C["accent"], "lw": 2}, ax=ax)
            r2, _ = stats.pearsonr(fdf["screen_time_before_sleep"], fdf["sleep_hours"])
            ax.set_title(f"Bedtime Screen Time → Sleep Quality  (r={r2:.3f})", fontsize=11)
            ax.set_xlabel("Screen Time Before Sleep (hours)"); ax.set_ylabel("Sleep Hours")
            _despine(ax); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()


# ═══════════════════════════════════════════════════
# TAB 3 — NOVEL FEATURES
# ═══════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">7 Domain-Informed Novel Features</div>',
                unsafe_allow_html=True)

    descriptions = {
        "digital_addiction_index":
            "**Digital Addiction Index (DAI)** — usage × stress ÷ sleep. Captures the compulsive-use pattern: high engagement under stress with poor recovery sleep.",
        "sleep_social_ratio":
            "**Sleep–Social Ratio** — sleep ÷ (usage + 0.1). A balance score: values >1 indicate healthy prioritisation of rest over scrolling.",
        "notification_pressure":
            "**Notification Pressure Score** — notifications × stress. Quantifies the cognitive interrupt load experienced under pressure.",
        "stress_caffeine_score":
            "**Stress–Caffeine Loop** — stress × log(caffeine + 1). Models the maladaptive coping cycle of stress → coffee → more stress.",
        "productivity_gap":
            "**Productivity Gap** — perceived score − actual score. Positive = overconfidence; negative = imposter syndrome.",
        "wellness_effective":
            "**Wellness Effectiveness Flag** — binary: has wellbeing app enabled AND stress below median. Identifies users for whom digital tools genuinely work.",
        "doomscroll_risk":
            "**Doomscrolling Risk Flag** — binary: top-quartile usage + top-quartile stress + bottom-quartile sleep simultaneously. Identifies the highest-risk behavioural cluster.",
    }

    NOVEL_COLS = [c for c in descriptions if c in fdf.columns]

    if NOVEL_COLS:
        n_cols = 3
        rows   = [NOVEL_COLS[i:i+n_cols] for i in range(0, len(NOVEL_COLS), n_cols)]

        for row in rows:
            cols = st.columns(len(row))
            for col, feat in zip(cols, row):
                with col:
                    st.markdown(f"""<div class="insight">{descriptions[feat]}</div>""",
                                unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(4.5, 3.5))
                    ax.scatter(fdf[feat], fdf["actual_productivity_score"],
                               alpha=0.3, s=12, color=C["primary"])
                    m, b = np.polyfit(fdf[feat], fdf["actual_productivity_score"], 1)
                    x_r  = np.linspace(fdf[feat].min(), fdf[feat].max(), 100)
                    ax.plot(x_r, m*x_r + b, color=C["danger"], lw=2)
                    corr = np.corrcoef(fdf[feat], fdf["actual_productivity_score"])[0, 1]
                    ax.set_title(f"r = {corr:.3f}", fontsize=10)
                    ax.set_xlabel(feat.replace("_", " ").title(), fontsize=8)
                    ax.set_ylabel("Productivity", fontsize=8)
                    _despine(ax); plt.tight_layout()
                    st.pyplot(fig, use_container_width=True); plt.close()

    # Doomscroll analysis
    st.markdown('<div class="section-header">Doomscrolling Risk × Wellbeing App Analysis</div>',
                unsafe_allow_html=True)

    if "doomscroll_risk" in fdf.columns and "has_digital_wellbeing_enabled" in fdf.columns:
        c1, c2 = st.columns(2)
        with c1:
            doom_prod = fdf.groupby(
                ["doomscroll_risk", "has_digital_wellbeing_enabled"]
            )["actual_productivity_score"].mean().unstack()
            doom_prod.index = ["No Risk", "Doomscroll Risk"]
            fig, ax = plt.subplots(figsize=(6, 4))
            x_d   = np.arange(len(doom_prod)); w = 0.35
            b_a = ax.bar(x_d - w/2, doom_prod.get(0, [0, 0]), w, color=C["danger"],   alpha=0.85, label="No Wellbeing App")
            b_b = ax.bar(x_d + w/2, doom_prod.get(1, [0, 0]), w, color=C["success"], alpha=0.85, label="Wellbeing App ON")
            ax.set_xticks(x_d); ax.set_xticklabels(doom_prod.index)
            ax.set_ylabel("Mean Productivity"); ax.set_ylim(0, 10); ax.legend(fontsize=9)
            ax.set_title("Wellbeing App Effect on Doomscrolling Risk", fontsize=11)
            for bars_grp in [b_a, b_b]:
                for bar in bars_grp:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.12,
                            f"{bar.get_height():.2f}", ha="center", fontsize=9, fontweight="bold")
            _despine(ax); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        with c2:
            ds_count = fdf["doomscroll_risk"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie(ds_count.values, labels=["No Risk", "At Risk"],
                   colors=[C["success"], C["danger"]], autopct="%1.1f%%",
                   startangle=140, wedgeprops=dict(edgecolor="white", linewidth=2))
            ax.set_title("Doomscroll Risk Distribution", fontsize=11)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()


# ═══════════════════════════════════════════════════
# TAB 4 — ML MODELS
# ═══════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Machine Learning Model Performance</div>',
                unsafe_allow_html=True)

    m = metrics
    if m:
        c1, c2, c3, c4 = st.columns(4)
        for col, label, value, color in [
            (c1, "RF R² Score",     f"{m.get('rf_r2', 0):.4f}",         C["primary"]),
            (c2, "Ridge R²",        f"{m.get('ridge_r2', 0):.4f}",      C["success"]),
            (c3, "Classifier Acc.", f"{m.get('clf_accuracy', 0)*100:.2f}%", C["purple"]),
            (c4, "Silhouette",      f"{m.get('silhouette', 0):.4f}",    C["accent"]),
        ]:
            with col:
                st.markdown(f"""
                <div class="kpi-card" style="border-color:{color}; text-align:center">
                  <div class="kpi-value" style="color:{color}; font-size:1.8rem">{value}</div>
                  <div class="kpi-label">{label}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Feature importance chart from file
    fi_path = "outputs/10_feature_importance.png"
    if os.path.exists(fi_path):
        with col1:
            st.markdown("**Top 20 Feature Importances — Random Forest**")
            st.image(fi_path, use_container_width=True)
    else:
        with col1:
            if "regressor" in models and "feature_cols" in models:
                rf_reg = models["regressor"]
                feat_cols = models["feature_cols"]
                importance = (
                    pd.DataFrame({"Feature": feat_cols, "Importance": rf_reg.feature_importances_})
                    .sort_values("Importance", ascending=False).head(20)
                )
                fig, ax = plt.subplots(figsize=(7, 6))
                sns.barplot(data=importance, x="Importance", y="Feature",
                            palette="Blues_r", ax=ax)
                ax.set_title("Top 20 Feature Importances", fontsize=12)
                _despine(ax); plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()

    # Confusion matrix
    cm_path = "outputs/11_confusion_matrix.png"
    if os.path.exists(cm_path):
        with col2:
            st.markdown("**Classification Confusion Matrix**")
            st.image(cm_path, use_container_width=True)

    # Persona clusters
    st.markdown('<div class="section-header">User Persona Clusters (K-Means, k=3)</div>',
                unsafe_allow_html=True)

    if "persona_cluster" in fdf.columns:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 5))
            palette = {0: C["success"], 1: C["danger"], 2: C["accent"]}
            labels  = {0: "Balanced", 1: "At-Risk", 2: "Moderate"}
            for cid in sorted(fdf["persona_cluster"].unique()):
                mask = fdf["persona_cluster"] == cid
                ax.scatter(fdf.loc[mask, "daily_social_media_time"],
                           fdf.loc[mask, "actual_productivity_score"],
                           alpha=0.5, s=20,
                           color=palette.get(cid, "#888"),
                           label=labels.get(cid, f"Cluster {cid}"))
            ax.set_xlabel("Daily Social Media Usage (hours)")
            ax.set_ylabel("Actual Productivity Score")
            ax.set_title("User Persona Clusters", fontsize=12)
            ax.legend(title="Persona"); _despine(ax); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        with c2:
            persona_summary = fdf.groupby("persona_cluster").agg(
                n=("actual_productivity_score", "count"),
                avg_productivity=("actual_productivity_score", "mean"),
                avg_usage=("daily_social_media_time", "mean"),
                avg_stress=("stress_level", "mean"),
                avg_sleep=("sleep_hours", "mean"),
            ).round(2)
            persona_summary.index = [labels.get(i, f"Cluster {i}") for i in persona_summary.index]
            persona_summary.columns = ["Count", "Avg Productivity", "Avg Usage (h)", "Avg Stress", "Avg Sleep"]
            st.markdown("**Persona Profile Summary**")
            st.dataframe(persona_summary, use_container_width=True)


# ═══════════════════════════════════════════════════
# TAB 5 — PREDICT
# ═══════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">🎯 Productivity Predictor</div>',
                unsafe_allow_html=True)
    st.markdown("Enter your profile below to get an AI-powered productivity score estimate.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📱 Social Media**")
            sm_time     = st.slider("Daily SM usage (hours)", 0.0, 12.0, 3.0, 0.5)
            notifs      = st.slider("Daily notifications",    0, 200, 50)
            screen_sleep= st.slider("Screen time before sleep (hours)", 0.0, 4.0, 1.0, 0.25)
            platform_choice = st.selectbox("Platform", ["Instagram", "Tiktok", "Twitter", "Youtube", "Facebook", "Reddit"])

        with col2:
            st.markdown("**😴 Wellbeing**")
            sleep_h     = st.slider("Sleep hours", 3.0, 10.0, 7.0, 0.5)
            stress      = st.slider("Stress level (1–10)", 1, 10, 5)
            coffee      = st.slider("Coffee cups/day", 0, 6, 2)
            offline_h   = st.slider("Weekly offline hours", 0.0, 40.0, 10.0, 1.0)

        with col3:
            st.markdown("**💼 Work**")
            work_h      = st.slider("Work hours/day", 1.0, 14.0, 8.0, 0.5)
            job_sat     = st.slider("Job satisfaction (1–10)", 1, 10, 6)
            breaks      = st.slider("Breaks during work/day", 0, 10, 3)
            focus_app   = st.selectbox("Uses focus apps?", ["Yes", "No"])
            wellbeing   = st.selectbox("Digital wellbeing app?", ["Yes", "No"])

        submit = st.form_submit_button("🔮 Predict My Productivity", use_container_width=True)

    if submit:
        # Build novel features
        focus_int    = 1 if focus_app    == "Yes" else 0
        wellbeing_int= 1 if wellbeing    == "Yes" else 0
        dai   = (sm_time * stress) / (sleep_h + 0.1)
        ssr   = sleep_h / (sm_time + 0.1)
        np_   = notifs * stress
        scs   = stress * np.log1p(coffee)
        we    = 1 if (wellbeing_int == 1 and stress < df["stress_level"].median()) else 0
        dr    = 1 if (sm_time > df["daily_social_media_time"].quantile(0.75) and
                      stress  > df["stress_level"].quantile(0.75) and
                      sleep_h < df["sleep_hours"].quantile(0.25)) else 0

        user_data_base = {
            "daily_social_media_time": sm_time, "number_of_notifications": notifs,
            "work_hours_per_day": work_h,        "stress_level": stress,
            "sleep_hours": sleep_h,              "screen_time_before_sleep": screen_sleep,
            "coffee_consumption_per_day": coffee,"weekly_offline_hours": offline_h,
            "job_satisfaction_score": job_sat,   "uses_focus_apps": focus_int,
            "has_digital_wellbeing_enabled": wellbeing_int, "breaks_during_work": breaks,
            "digital_addiction_index": dai,      "sleep_social_ratio": ssr,
            "notification_pressure": np_,        "stress_caffeine_score": scs,
            "productivity_gap": 0,               "wellness_effective": we,
            "doomscroll_risk": dr,               "age_group_encoded": 1,
        }

        # Tier label
        tier_labels = {0: "Low", 1: "Medium", 2: "High"}
        tier_colors = {"Low": C["danger"], "Medium": C["accent"], "High": C["success"]}

        if "regressor" in models and "feature_cols" in models:
            feat_cols = models["feature_cols"]
            user_row  = pd.DataFrame([{c: user_data_base.get(c, 0) for c in feat_cols}])
            pred_score = float(models["regressor"].predict(user_row)[0])
            pred_score = np.clip(pred_score, 0, 10)

            if pred_score >= 7:
                tier = "High"; tier_color = C["success"]
            elif pred_score >= 4:
                tier = "Medium"; tier_color = C["accent"]
            else:
                tier = "Low"; tier_color = C["danger"]

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(f"""
                <div class="pred-box" style="background:{tier_color}15; border: 2px solid {tier_color}">
                  <div class="pred-score" style="color:{tier_color}">{pred_score:.2f}</div>
                  <div class="pred-tier" style="color:{tier_color}">Predicted Productivity</div>
                  <div style="color:#6C757D; font-size:0.82rem; margin-top:0.3rem">out of 10.0</div>
                </div>""", unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="pred-box" style="background:#F8F9FA; border: 2px solid {tier_color}">
                  <div class="pred-score" style="color:{tier_color}">{tier}</div>
                  <div class="pred-tier" style="color:#6C757D">Productivity Tier</div>
                  <div style="font-size:0.85rem; margin-top:0.4rem; color:#6C757D">Low / Medium / High</div>
                </div>""", unsafe_allow_html=True)

            with c3:
                risk_level = "🔴 High Risk" if dr else ("🟡 Moderate" if stress > 6 else "🟢 Low Risk")
                st.markdown(f"""
                <div class="pred-box" style="background:#F8F9FA; border: 2px solid #6C757D">
                  <div class="pred-score" style="color:#2C3E50; font-size:1.8rem">DAI {dai:.2f}</div>
                  <div class="pred-tier" style="color:#6C757D">Addiction Index</div>
                  <div style="font-size:0.9rem; margin-top:0.4rem">{risk_level}</div>
                </div>""", unsafe_allow_html=True)

            # Personalised insights
            st.markdown('<div class="section-header">Personalised Recommendations</div>',
                        unsafe_allow_html=True)

            insights = []
            if sm_time > 4:
                insights.append(("🔴", f"Your {sm_time}h daily usage exceeds the 4h tipping point. Consider setting a daily limit."))
            if sleep_h < 7:
                insights.append(("🟡", f"Only {sleep_h}h of sleep is below the 7h optimal threshold — this amplifies SM harms."))
            if dr:
                insights.append(("🔴", "You show the doomscrolling risk pattern. High usage + high stress + low sleep is the most damaging combination."))
            if wellbeing_int == 0:
                insights.append(("🟢", "Enabling a digital wellbeing app can boost productivity by ~12% for at-risk users."))
            if focus_int == 0:
                insights.append(("🟢", "Focus apps significantly reduce notification pressure. Consider Forest, Freedom, or Cold Turkey."))
            if stress > 7:
                insights.append(("🟡", "High stress amplifies the negative effect of social media on productivity."))
            if not insights:
                insights.append(("🟢", "Your profile looks relatively balanced! Keep monitoring usage and sleep quality."))

            for icon, msg in insights:
                bg = "#FFF0F0" if icon == "🔴" else "#FFFBF0" if icon == "🟡" else "#F0FAF8"
                border = C["danger"] if icon == "🔴" else C["accent"] if icon == "🟡" else C["success"]
                st.markdown(f"""
                <div style="background:{bg}; border-left:4px solid {border}; border-radius:0 8px 8px 0;
                     padding:0.8rem 1rem; margin:0.4rem 0; font-size:0.88rem">
                  {icon} {msg}
                </div>""", unsafe_allow_html=True)
        else:
            st.info("🔧 Run the full pipeline first (01–05) to load trained models.")
            st.markdown("**Estimated score based on heuristics:**")
            heuristic = 5.0
            if sm_time > 4: heuristic -= 1.2
            if sleep_h < 6: heuristic -= 0.8
            if stress > 7:  heuristic -= 0.7
            if focus_int:   heuristic += 0.5
            heuristic = np.clip(heuristic, 1, 10)
            st.metric("Heuristic Productivity Estimate", f"{heuristic:.1f} / 10")


# ═══════════════════════════════════════════════════
# TAB 6 — STORY
# ═══════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">📖 The Data Story — Does Social Media Harm Productivity?</div>',
                unsafe_allow_html=True)

    # Master dashboard image
    story_path = "outputs/13_story_dashboard.png"
    if os.path.exists(story_path):
        st.image(story_path, caption="Complete Analysis Dashboard", use_container_width=True)
    else:
        st.info("Run `06_storytelling.py` to generate the master dashboard image.")

    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        ### 🔍 The Finding
        **YES — social media negatively impacts productivity, but conditionally.**

        The relationship is non-linear and mediated by sleep and stress.
        """)
    with c2:
        st.markdown("""
        ### 🔗 The Mechanism
        High usage → **sleep sacrifice** → elevated stress → **cognitive impairment** → lower output.

        The tipping point is approximately **4 hours/day**.
        """)
    with c3:
        st.markdown("""
        ### 💡 The Intervention
        Wellbeing apps provide a **measurable +12% buffer** for at-risk users.

        Short-form platforms (TikTok, Instagram) carry the **highest risk**.
        """)

    r_val, _ = stats.pearsonr(
        fdf["daily_social_media_time"], fdf["actual_productivity_score"]
    )
    st.markdown("---")
    st.markdown(f"""
    <div class="insight">
      <strong>Final Statistical Answer:</strong> Pearson r = {r_val:.3f} between daily usage
      and productivity score across {len(fdf):,} participants.
      The DAI (Digital Addiction Index) is the single strongest predictor in the Random Forest model,
      confirming that the combined effect of usage, stress, and sleep deprivation drives outcomes
      more than any single factor alone.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style="text-align:center; color:#6C757D; font-size:0.82rem; padding:0.5rem">
  Social Media vs Productivity Analysis · MSc Data Science ·
  Built with Python, scikit-learn, matplotlib & Streamlit
</div>
""", unsafe_allow_html=True)
