"""
╔══════════════════════════════════════════════════════════════╗
║   SOCIAL MEDIA VS PRODUCTIVITY  |  STEP 3: EDA              ║
║   Every chart uses a DISTINCT, RICH colour scheme           ║
╚══════════════════════════════════════════════════════════════╝
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
import os, warnings
warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# ══════════════════════════════════════════════════════════════
# GLOBAL COLOUR SYSTEM
# ══════════════════════════════════════════════════════════════
C = {
    "teal":   "#1A9E8F", "teal_lt":   "#C8EDE9",
    "red":    "#E63946", "red_lt":    "#FFCAD4",
    "purple": "#7B5EA7", "purple_lt": "#DDD4F0",
    "coral":  "#D85A30", "coral_lt":  "#FAD4C0",
    "blue":   "#1A6B8A", "blue_lt":   "#BDDCEE",
    "gold":   "#D4A017", "gold_lt":   "#FFF0C0",
    "green":  "#2D9B5A", "green_lt":  "#C8EDD8",
    "orange": "#F4A261", "orange_lt": "#FDE8D0",
    "pink":   "#C2478B", "pink_lt":   "#F5D0E8",
    "navy":   "#0F1B35",
    "white":  "#FFFFFF",
    "bg":     "#F7F8FA",
    "muted":  "#6C757D",
}

# One distinct colour per key metric — used across ALL charts
METRIC_C = {
    "actual_productivity_score":  (C["teal"],   C["teal_lt"]),
    "daily_social_media_time":    (C["red"],    C["red_lt"]),
    "sleep_hours":                (C["purple"], C["purple_lt"]),
    "stress_level":               (C["coral"],  C["coral_lt"]),
    "job_satisfaction_score":     (C["blue"],   C["blue_lt"]),
    "coffee_consumption_per_day": (C["gold"],   C["gold_lt"]),
    "number_of_notifications":    (C["pink"],   C["pink_lt"]),
    "work_hours_per_day":         (C["green"],  C["green_lt"]),
    "screen_time_before_sleep":   (C["orange"], C["orange_lt"]),
}

# Distinct platform colours
PLATFORM_C = ["#2D9B5A","#1A6B8A","#7B5EA7","#D4A017","#F4A261","#E63946"]

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  C["white"],
    "axes.titleweight":  "bold",
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8.5,
    "legend.framealpha": 0.9,
})

def style_ax(ax, spine_color=C["muted"], bg=C["bg"]):
    ax.set_facecolor(bg)
    for sp in ["left","bottom"]:
        ax.spines[sp].set_color(spine_color)
        ax.spines[sp].set_linewidth(1.3)

def add_stats_box(ax, data, color):
    txt = f"n={len(data):,}\nμ={data.mean():.2f}\nσ={data.std():.2f}\nskew={data.skew():.2f}"
    ax.text(0.97,0.97,txt, transform=ax.transAxes, fontsize=7.5,
            va="top", ha="right", color=C["muted"],
            bbox=dict(boxstyle="round,pad=0.35",facecolor=C["white"],
                      edgecolor=color, linewidth=0.9, alpha=0.92))

# ── Load ─────────────────────────────────────────────────────
df = pd.read_csv("data/cleaned_social_media.csv")
for col in ["gender","job_type","social_platform_preference"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()
df["age_group"] = pd.cut(df["age"],bins=[18,25,35,45,55,100],
                          labels=["18-25","26-35","36-45","46-55","56+"])
df["usage_band"] = pd.cut(df["daily_social_media_time"],bins=[0,1,2,4,6,20],
                           labels=["<1h","1-2h","2-4h","4-6h",">6h"])
print(f"Loaded: {df.shape}")


# ════════════════════════════════════════════════════════════════
# CHART 1 — UNIVARIATE DISTRIBUTIONS  (6 distinct colours)
# ════════════════════════════════════════════════════════════════
NUM_COLS = [
    "actual_productivity_score","daily_social_media_time",
    "sleep_hours",              "stress_level",
    "job_satisfaction_score",   "coffee_consumption_per_day",
]
TITLES = {
    "actual_productivity_score":  "Actual Productivity Score",
    "daily_social_media_time":    "Daily Social Media Time (hrs)",
    "sleep_hours":                "Sleep Hours",
    "stress_level":               "Stress Level",
    "job_satisfaction_score":     "Job Satisfaction Score",
    "coffee_consumption_per_day": "Coffee Consumption / Day",
}

fig, axes = plt.subplots(2,3,figsize=(18,10))
fig.suptitle("Univariate Distributions — Key Metrics",
             fontsize=18, fontweight="bold", y=1.02, color=C["navy"])

for ax, col in zip(axes.flat, NUM_COLS):
    dark, light = METRIC_C[col]
    data = df[col].dropna()
    mean_v, med_v, std_v = data.mean(), data.median(), data.std()

    # Tinted subplot background
    ax.set_facecolor(light + "55")

    # Histogram + KDE in the metric's own colour
    sns.histplot(data, kde=True, ax=ax,
                 color=dark, alpha=0.72,
                 edgecolor="white", linewidth=0.5,
                 line_kws={"color": dark, "lw": 2.5, "alpha": 1.0})

    # Mean line — white stroke + dark colour so it pops on any bg
    ax.axvline(mean_v, color=C["red"], lw=2.2, ls="--", zorder=6,
               label=f"Mean  {mean_v:.2f}")
    ax.axvline(med_v,  color=C["navy"], lw=2.0, ls=":", zorder=6,
               label=f"Median {med_v:.2f}")

    # ±1σ shaded band
    ax.axvspan(mean_v-std_v, mean_v+std_v, alpha=0.13, color=dark, zorder=1)

    # Coloured title + spines
    ax.set_title(TITLES[col], color=dark, fontsize=12, fontweight="bold", pad=7)
    ax.set_xlabel(""); ax.set_ylabel("Count", fontsize=9)
    for sp in ["left","bottom"]:
        ax.spines[sp].set_color(dark); ax.spines[sp].set_linewidth(1.4)

    # Stats annotation
    add_stats_box(ax, data, dark)

    ax.legend(frameon=True, edgecolor=dark, facecolor=C["white"],
              fontsize=8.5, loc="upper left")

plt.tight_layout(pad=2.5)
plt.savefig("outputs/01_univariate.png", dpi=300, bbox_inches="tight")
plt.close(); print(" 01_univariate.png")


# ════════════════════════════════════════════════════════════════
# CHART 2 — CORRELATION HEATMAP
# ════════════════════════════════════════════════════════════════
numeric_df = df.select_dtypes(include=np.number)
# Drop indicator columns
indicator_cols = [c for c in numeric_df.columns if "_was_missing" in c]
numeric_df = numeric_df.drop(columns=indicator_cols, errors="ignore")
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(15,12))
fig.patch.set_facecolor(C["white"])
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = LinearSegmentedColormap.from_list("rdb",[C["red"],"#FAFAFA",C["blue"]],N=256)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0.6, linecolor="#E0E0E0",
            annot_kws={"size":7.5}, ax=ax,
            cbar_kws={"shrink":0.7,"label":"Pearson r"})
ax.set_title("Feature Correlation Matrix — Lower Triangle",
             fontsize=15, fontweight="bold", pad=14, color=C["navy"])
ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig("outputs/02_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(); print(" 02_correlation_heatmap.png")


# ════════════════════════════════════════════════════════════════
# CHART 3 — PLATFORM vs PRODUCTIVITY  (6 distinct colours)
# ════════════════════════════════════════════════════════════════
platform_order = (df.groupby("social_platform_preference")
                    ["actual_productivity_score"]
                    .median().sort_values(ascending=False).index.tolist())
n_plat = len(platform_order)
pal    = PLATFORM_C[:n_plat]

fig, axes = plt.subplots(1,2,figsize=(17,7))
fig.suptitle("Platform vs Productivity", fontsize=16, fontweight="bold", color=C["navy"])

# Boxplot — each platform its own colour
sns.boxplot(data=df, x="social_platform_preference",
            y="actual_productivity_score",
            order=platform_order, palette=pal, ax=axes[0],
            linewidth=1.3,
            flierprops=dict(marker="o",markersize=3,alpha=0.35))
axes[0].set_title("Productivity Score Distribution by Platform", color=C["navy"])
axes[0].set_xlabel("Platform"); axes[0].set_ylabel("Actual Productivity Score")
axes[0].tick_params(axis="x", rotation=30); style_ax(axes[0])

# Horizontal bar — coloured + annotated
usage_means = (df.groupby("social_platform_preference")
                 ["daily_social_media_time"].mean().loc[platform_order])
bars = axes[1].barh(range(len(usage_means)), usage_means.values,
                    color=pal, edgecolor="white", linewidth=0.8,
                    height=0.62, alpha=0.92)
axes[1].set_yticks(range(len(usage_means)))
axes[1].set_yticklabels(usage_means.index, fontsize=10)
for i,(bar,val) in enumerate(zip(bars,usage_means.values)):
    axes[1].text(val+0.04, bar.get_y()+bar.get_height()/2,
                 f"{val:.2f}h", va="center", fontsize=9.5, fontweight="bold")
axes[1].set_title("Avg Daily Usage by Platform", color=C["navy"])
axes[1].set_xlabel("Hours / Day"); style_ax(axes[1])

plt.tight_layout()
plt.savefig("outputs/03_platform_productivity.png", dpi=300, bbox_inches="tight")
plt.close(); print(" 03_platform_productivity.png")


# ════════════════════════════════════════════════════════════════
# CHART 4 — USAGE vs PRODUCTIVITY  (zoned scatter + coloured bars)
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1,2,figsize=(17,7))
fig.suptitle("Social Media Usage vs Productivity",
             fontsize=16, fontweight="bold", color=C["navy"])

ax = axes[0]
# Zone shading
ax.axvspan(0,                                    2,        alpha=0.14, color=C["green"],  label="Safe (<2h)")
ax.axvspan(2,                                    4,        alpha=0.14, color=C["gold"],   label="Caution (2-4h)")
ax.axvspan(4, df["daily_social_media_time"].max()+0.5,  alpha=0.16, color=C["red"],    label="Risk (>4h)")
# Scatter + LOWESS
ax.scatter(df["daily_social_media_time"], df["actual_productivity_score"],
           c=df["stress_level"], cmap="RdYlGn_r",
           alpha=0.25, s=16, zorder=3)
# Regression line via polyfit (avoids lowess deprecation)
x_ = df["daily_social_media_time"]; y_ = df["actual_productivity_score"]
from scipy.interpolate import make_interp_spline
bins_  = np.linspace(x_.min(), x_.max(), 22)
bx_ = (bins_[:-1]+bins_[1:])/2
by_ = [y_[(x_>=lo)&(x_<hi)].mean() for lo,hi in zip(bins_[:-1],bins_[1:])]
valid_ = [not np.isnan(v) for v in by_]
bx2,by2 = np.array(bx_)[valid_], np.array(by_)[valid_]
try:
    spl = make_interp_spline(bx2,by2,k=3)
    xs_ = np.linspace(bx2.min(),bx2.max(),200)
    ax.plot(xs_, spl(xs_), color=C["red"], lw=2.8, zorder=5, label="Smoothed trend")
except:
    ax.plot(bx2,by2, color=C["red"], lw=2.8, zorder=5)
r,p = stats.pearsonr(df["daily_social_media_time"], df["actual_productivity_score"])
ax.set_title(f"Usage vs Productivity  (r={r:.3f}, p={p:.4f})", color=C["navy"])
ax.set_xlabel("Daily Social Media Time (hours)")
ax.set_ylabel("Actual Productivity Score")
ax.legend(fontsize=8.5, loc="upper right")
style_ax(ax)

# Colour-coded band bars
ax2 = axes[1]
grouped = df.groupby("usage_band", observed=True)["actual_productivity_score"].mean()
band_c  = [C["green"], C["teal"], C["gold"], C["orange"], C["red"]]
bars2   = ax2.bar(grouped.index, grouped.values,
                  color=band_c, edgecolor="white", linewidth=0.9,
                  alpha=0.93, width=0.62)
for bar,val in zip(bars2, grouped.values):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.08,
             f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
ax2.set_title("Mean Productivity by Usage Band", color=C["navy"])
ax2.set_xlabel("Usage Band"); ax2.set_ylabel("Mean Productivity Score")
ax2.set_ylim(0,11)
patches = [mpatches.Patch(color=c, label=l) for c,l in
           zip(band_c,["<1h (safe)","1-2h","2-4h","4-6h",">6h (risk)"])]
ax2.legend(handles=patches, fontsize=8)
style_ax(ax2)

plt.tight_layout()
plt.savefig("outputs/04_usage_vs_productivity.png", dpi=300, bbox_inches="tight")
plt.close(); print(" 04_usage_vs_productivity.png")


# ════════════════════════════════════════════════════════════════
# CHART 5 — LIFESTYLE FACTORS  (3 distinct scatter colours)
# ════════════════════════════════════════════════════════════════
LIFE = [
    ("sleep_hours",            "Sleep Hours",      C["purple"], C["purple_lt"], C["red"]),
    ("stress_level",           "Stress Level",     C["coral"],  C["coral_lt"],  C["navy"]),
    ("job_satisfaction_score", "Job Satisfaction", C["blue"],   C["blue_lt"],   C["orange"]),
]

fig, axes = plt.subplots(1,3,figsize=(18,7))
fig.suptitle("Lifestyle Factors vs Productivity",
             fontsize=16, fontweight="bold", color=C["navy"])

for ax,(xcol,title,sc,bg,lc) in zip(axes,LIFE):
    ax.set_facecolor(bg+"55")
    sns.regplot(data=df, x=xcol, y="actual_productivity_score",
                scatter_kws={"alpha":0.22,"s":18,"color":sc},
                line_kws={"color":lc,"lw":2.8}, ax=ax)
    r,_ = stats.pearsonr(df[xcol], df["actual_productivity_score"])
    ax.set_title(f"{title}  (r = {r:.3f})", color=sc, fontsize=12, fontweight="bold")
    ax.set_ylabel("Productivity Score"); ax.set_xlabel(title)
    for sp in ["left","bottom"]:
        ax.spines[sp].set_color(sc); ax.spines[sp].set_linewidth(1.4)
    # Annotate r text
    ax.text(0.03,0.97,f"r = {r:.3f}", transform=ax.transAxes,
            fontsize=10, va="top", fontweight="bold", color=lc,
            bbox=dict(boxstyle="round,pad=0.3",facecolor=C["white"],
                      edgecolor=lc, linewidth=1, alpha=0.9))

plt.tight_layout(pad=2.0)
plt.savefig("outputs/05_lifestyle_factors.png", dpi=300, bbox_inches="tight")
plt.close(); print(" 05_lifestyle_factors.png")


# ════════════════════════════════════════════════════════════════
# CHART 6 — DEMOGRAPHICS  (Set2 + Spectral)
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1,2,figsize=(17,7))
fig.suptitle("Demographics Analysis",
             fontsize=16, fontweight="bold", color=C["navy"])

sns.barplot(data=df, x="age_group", y="actual_productivity_score",
            hue="gender", palette="Set2", ax=axes[0],
            errorbar="se", edgecolor="white", linewidth=0.8)
axes[0].set_title("Productivity by Age Group & Gender", color=C["navy"])
axes[0].set_xlabel("Age Group"); axes[0].set_ylabel("Mean Productivity Score")
axes[0].legend(title="Gender", fontsize=9)
style_ax(axes[0])

sns.boxplot(data=df, x="age_group", y="daily_social_media_time",
            palette="Spectral", ax=axes[1], linewidth=1.1,
            flierprops=dict(marker="o",markersize=3,alpha=0.4))
axes[1].set_title("Social Media Usage by Age Group", color=C["navy"])
axes[1].set_xlabel("Age Group"); axes[1].set_ylabel("Daily Usage (hours)")
style_ax(axes[1])

plt.tight_layout()
plt.savefig("outputs/06_demographics.png", dpi=300, bbox_inches="tight")
plt.close(); print(" 06_demographics.png")


# ════════════════════════════════════════════════════════════════
# CHART 7 — NOTIFICATIONS vs PRODUCTIVITY  (gradient scatter)
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11,7))
style_ax(ax, spine_color=C["pink"])
sc = ax.scatter(df["number_of_notifications"],
                df["actual_productivity_score"],
                c=df["number_of_notifications"], cmap="RdYlGn_r",
                alpha=0.28, s=18, zorder=2)
cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.01)
cbar.set_label("Notification Count", fontsize=9)
m,b = np.polyfit(df["number_of_notifications"],df["actual_productivity_score"],1)
xr = np.linspace(df["number_of_notifications"].min(),
                 df["number_of_notifications"].max(),200)
ax.plot(xr, m*xr+b, color=C["red"], lw=2.8, zorder=4, label="Trend line")
r,p = stats.pearsonr(df["number_of_notifications"],df["actual_productivity_score"])
ax.set_title(f"Notification Interruptions vs Productivity  (r={r:.3f}, p={p:.4f})",
             fontsize=13, fontweight="bold", color=C["navy"])
ax.set_xlabel("Number of Daily Notifications")
ax.set_ylabel("Actual Productivity Score")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("outputs/07_notifications_vs_productivity.png", dpi=300, bbox_inches="tight")
plt.close(); print(" 07_notifications_vs_productivity.png")

print("\n ALL EDA CHARTS COMPLETE")
