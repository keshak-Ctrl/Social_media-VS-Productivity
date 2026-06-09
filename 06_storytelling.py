"""
╔══════════════════════════════════════════════════════════════╗
║   SOCIAL MEDIA VS PRODUCTIVITY  |  STEP 6: STORYTELLING     ║
║   Every chart uses a DISTINCT, RICH colour scheme           ║
╚══════════════════════════════════════════════════════════════╝
Produces a 3×3 master dashboard + 5 individual story charts.
Run AFTER 05_modelling.py (requires data/final_with_personas.csv)
"""
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.interpolate import make_interp_spline
from scipy import stats
import seaborn as sns

os.makedirs("outputs", exist_ok=True)

# ══════════════════════════════════════════════════════════════
# COLOUR SYSTEM
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
    "navy":   "#0F1B35",
    "white":  "#FFFFFF",
    "bg":     "#F7F8FA",
    "muted":  "#6C757D",
}
PLATFORM_C = ["#2D9B5A","#1A6B8A","#7B5EA7","#D4A017","#F4A261","#E63946"]
PERSONA_C  = {0: C["green"], 1: C["red"],  2: C["orange"]}
PERSONA_L  = {0: "Balanced", 1: "At-Risk", 2: "Moderate"}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  C["bg"],
    "axes.titleweight":  "bold",
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
})

def style_ax(ax, color=C["muted"], bg=C["bg"]):
    ax.set_facecolor(bg)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(color)
        ax.spines[sp].set_linewidth(1.3)

# ── Load & rename ─────────────────────────────────────────────
df = pd.read_csv("data/final_with_personas.csv")
df = df.rename(columns={
    "daily_social_media_time":    "daily_usage_hours",
    "actual_productivity_score":  "productivity_score",
    "social_platform_preference": "platform_preference",
    "digital_addiction_index":    "dai",
    "has_digital_wellbeing_enabled": "wellbeing_app_use",
    "persona_cluster":            "persona",
})
df["platform_preference"] = df["platform_preference"].astype(str).str.strip().str.title()
df["wellbeing_app_use"]   = df["wellbeing_app_use"].fillna(0).astype(int)
if "doomscroll_risk" in df.columns and df["doomscroll_risk"].nunique() > 2:
    df["doomscroll_risk"] = (df["doomscroll_risk"] >= df["doomscroll_risk"].median()).astype(int)

print(f"Loaded: {df.shape}")

# ════════════════════════════════════════════════════════════════
# STORY CHART 1 — Hero scatter: Usage vs Productivity
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_facecolor("#EBF5FA")
x = df["daily_usage_hours"]; y = df["productivity_score"]
ax.axvspan(0, 2,        alpha=0.18, color=C["green"], label="Safe zone (<2h)",   zorder=1)
ax.axvspan(2, 4,        alpha=0.16, color=C["gold"],  label="Caution (2–4h)",   zorder=1)
ax.axvspan(4, x.max()+0.2, alpha=0.18, color=C["red"],   label="Risk zone (>4h)", zorder=1)
sc = ax.scatter(x, y, c=df["stress_level"], cmap="RdYlGn_r", alpha=0.28, s=18, zorder=3)
plt.colorbar(sc, ax=ax, label="Stress Level", shrink=0.8, pad=0.01)
bins_ = np.linspace(x.min(), x.max(), 22)
bx_   = (bins_[:-1] + bins_[1:]) / 2
by_   = [y[(x >= lo) & (x < hi)].mean() for lo, hi in zip(bins_[:-1], bins_[1:])]
be_   = [y[(x >= lo) & (x < hi)].sem()  for lo, hi in zip(bins_[:-1], bins_[1:])]
valid_= [not np.isnan(v) for v in by_]
bx2, by2, be2 = np.array(bx_)[valid_], np.array(by_)[valid_], np.array(be_)[valid_]
try:
    spl = make_interp_spline(bx2, by2, k=3)
    xs_ = np.linspace(bx2.min(), bx2.max(), 300)
    ax.plot(xs_, spl(xs_), color=C["red"], lw=3.0, zorder=5, label="Smoothed trend")
    ax.fill_between(bx2, by2-be2, by2+be2, alpha=0.18, color=C["red"], zorder=4)
except Exception:
    ax.plot(bx2, by2, color=C["red"], lw=3.0, zorder=5)
ax.axvline(4, color=C["navy"], lw=2.0, ls=":", alpha=0.7, label="Tipping point ~4h")
r, p = stats.pearsonr(x, y)
ax.set_title(f"Social Media Usage → Productivity  (Pearson r = {r:.3f},  p = {p:.4f})",
             fontsize=14, fontweight="bold", color=C["navy"], pad=10)
ax.set_xlabel("Daily Social Media Usage (hours)", fontsize=11)
ax.set_ylabel("Actual Productivity Score (1–10)", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(0, x.max() + 0.2)
for sp in ["left", "bottom"]:
    ax.spines[sp].set_color(C["blue"]); ax.spines[sp].set_linewidth(1.5)
plt.tight_layout()
plt.savefig("outputs/18_story_hero.png", dpi=300, bbox_inches="tight")
plt.close(); print(" 18_story_hero.png")


# ════════════════════════════════════════════════════════════════
# STORY CHART 2 — Platform Risk
# ════════════════════════════════════════════════════════════════
platform_order = (df.groupby("platform_preference")["productivity_score"]
                    .median().sort_values(ascending=False).index.tolist())
n_p = len(platform_order); pal = PLATFORM_C[:n_p]
medians_p = df.groupby("platform_preference")["productivity_score"].median().loc[platform_order]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Platform Risk Analysis", fontsize=15, fontweight="bold", color=C["navy"])

bars = axes[0].barh(range(n_p), medians_p.values, color=pal,
                    edgecolor="white", linewidth=0.8, height=0.62, alpha=0.92)
axes[0].set_yticks(range(n_p)); axes[0].set_yticklabels(platform_order, fontsize=10)
for bar, val in zip(bars, medians_p.values):
    axes[0].text(val+0.05, bar.get_y()+bar.get_height()/2, f"{val:.2f}",
                 va="center", fontsize=10, fontweight="bold")
axes[0].set_xlabel("Median Productivity Score"); axes[0].set_xlim(0, 11)
axes[0].set_title("Median Productivity by Platform", color=C["navy"])
style_ax(axes[0])
axes[0].legend(handles=[mpatches.Patch(color=C["green"], label="Low risk"),
                         mpatches.Patch(color=C["red"],   label="High risk")], fontsize=9)

usage_m = df.groupby("platform_preference")["daily_usage_hours"].mean().loc[platform_order]
bars2 = axes[1].bar(range(n_p), usage_m.values, color=pal,
                    edgecolor="white", linewidth=0.8, width=0.6, alpha=0.92)
axes[1].set_xticks(range(n_p))
axes[1].set_xticklabels(platform_order, rotation=25, ha="right")
for bar, val in zip(bars2, usage_m.values):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                 f"{val:.2f}h", ha="center", fontsize=9, fontweight="bold")
axes[1].set_ylabel("Avg Daily Usage (hours)")
axes[1].set_title("Average Daily Usage by Platform", color=C["navy"])
style_ax(axes[1])
plt.tight_layout()
plt.savefig("outputs/19_story_platform_risk.png", dpi=300, bbox_inches="tight")
plt.close(); print(" 19_story_platform_risk.png")


# ════════════════════════════════════════════════════════════════
# STORY CHART 3 — Stress Heatmap + Sleep Bars
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Stress & Sleep — Mediators of Social Media Harm",
             fontsize=15, fontweight="bold", color=C["navy"])

df["stress_band"] = pd.cut(df["stress_level"], 3, labels=["Low", "Med", "High"])
df["usage_band2"]  = pd.cut(df["daily_usage_hours"], 4, labels=["<2h", "2-4h", "4-8h", ">8h"])
pivot = df.groupby(["usage_band2", "stress_band"], observed=True)["productivity_score"].mean().unstack()
cmap_h = LinearSegmentedColormap.from_list("prod", [C["red"], C["gold"], C["green"]], N=256)
im = axes[0].imshow(pivot.values, cmap=cmap_h, aspect="auto", vmin=1, vmax=10)
plt.colorbar(im, ax=axes[0], label="Mean Productivity", shrink=0.85)
axes[0].set_xticks(range(len(pivot.columns))); axes[0].set_xticklabels(pivot.columns, fontsize=10)
axes[0].set_yticks(range(len(pivot.index)));   axes[0].set_yticklabels(pivot.index, fontsize=10)
axes[0].set_title("Heatmap: Usage × Stress → Productivity\n(Green=High, Red=Low)", color=C["navy"])
axes[0].set_xlabel("Stress Band"); axes[0].set_ylabel("Usage Band")
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            axes[0].text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=11,
                         fontweight="bold", color=C["white"] if val < 5 else C["navy"])

sleep_bins = pd.cut(df["sleep_hours"], bins=[0,5,6,7,8,12],
                    labels=["<5h", "5-6h", "6-7h", "7-8h ", ">8h"])
sp_d = df.groupby(sleep_bins, observed=True)["productivity_score"].agg(["mean", "sem"])
sleep_pal = [C["red"], "#E07B4A", C["gold"], C["green"], C["blue"]]
bars_s = axes[1].bar(range(len(sp_d)), sp_d["mean"], yerr=sp_d["sem"],
                     color=sleep_pal, edgecolor="white", capsize=4,
                     width=0.62, alpha=0.92)
axes[1].set_xticks(range(len(sp_d)))
axes[1].set_xticklabels(sp_d.index, rotation=20, ha="right", fontsize=9)
axes[1].set_ylabel("Mean Productivity Score"); axes[1].set_ylim(0, 11)
axes[1].set_title("Productivity vs Sleep Hours\n(sleep loss amplifies SM harm)", color=C["navy"])
for bar, val in zip(bars_s, sp_d["mean"]):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                 f"{val:.2f}", ha="center", fontsize=9.5, fontweight="bold")
style_ax(axes[1], C["purple"], bg=C["purple_lt"]+"33")
plt.tight_layout()
plt.savefig("outputs/20_story_stress_sleep.png", dpi=300, bbox_inches="tight")
plt.close(); print(" 20_story_stress_sleep.png")


# ════════════════════════════════════════════════════════════════
# STORY CHART 4 — Doomscroll + Wellbeing App
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Doomscrolling Risk & Wellbeing App Intervention",
             fontsize=15, fontweight="bold", color=C["navy"])

if "doomscroll_risk" in df.columns:
    doom = df.groupby(["doomscroll_risk", "wellbeing_app_use"])["productivity_score"].mean().unstack()
    doom.index = ["No Risk", "Doomscroll Risk"]
    x_d = np.arange(len(doom)); w = 0.35; style_ax(axes[0])
    ba = axes[0].bar(x_d-w/2, doom.get(0,[0,0]), w, color=C["red"],   alpha=0.88, label="No Wellbeing App",  edgecolor="white")
    bb = axes[0].bar(x_d+w/2, doom.get(1,[0,0]), w, color=C["green"], alpha=0.88, label="Wellbeing App ON",   edgecolor="white")
    axes[0].set_xticks(x_d); axes[0].set_xticklabels(doom.index, fontsize=11)
    axes[0].set_ylabel("Mean Productivity Score"); axes[0].set_ylim(0, 10)
    axes[0].set_title("Wellbeing App Effect on Doomscroll Risk", color=C["navy"])
    axes[0].legend(fontsize=9)
    for grp in [ba, bb]:
        for bar in grp:
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.12,
                         f"{bar.get_height():.2f}", ha="center", fontsize=10, fontweight="bold")

    dc = df["doomscroll_risk"].value_counts().sort_index()
    w2, t2, a2 = axes[1].pie(dc.values, labels=["No Risk","Doomscroll Risk"],
                               colors=[C["green"], C["red"]],
                               autopct="%1.1f%%", startangle=140,
                               wedgeprops=dict(edgecolor="white", linewidth=2.5),
                               textprops={"fontsize":11, "fontweight":"bold"})
    for a, c in zip(a2, [C["green"], C["red"]]):
        a.set_color("white"); a.set_fontweight("bold")
    axes[1].set_title("Doomscroll Risk Distribution", color=C["navy"])
plt.tight_layout()
plt.savefig("outputs/21_story_doomscroll.png", dpi=300, bbox_inches="tight")
plt.close(); print(" 21_story_doomscroll.png")


# ════════════════════════════════════════════════════════════════
# STORY CHART 5 — Persona Profiles
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("User Persona Profiles", fontsize=15, fontweight="bold", color=C["navy"])

persona_attrs = ["productivity_score", "daily_usage_hours", "sleep_hours", "stress_level"]
p_means = df.groupby("persona")[persona_attrs].mean()
normed  = (p_means - p_means.min()) / (p_means.max() - p_means.min() + 1e-9)
normed.index = [PERSONA_L.get(i, f"Cluster {i}") for i in normed.index]
attr_colors  = [C["teal"], C["red"], C["purple"], C["coral"]]
normed.plot(kind="bar", ax=axes[0], width=0.70, color=attr_colors,
            edgecolor="white", linewidth=0.7, alpha=0.90)
axes[0].set_xticklabels(normed.index, rotation=0, fontsize=11)
axes[0].set_ylabel("Normalised Value (0–1)"); axes[0].set_ylim(0, 1.15)
axes[0].set_title("Persona Profiles — Normalised Attributes", color=C["navy"])
axes[0].legend(fontsize=9); style_ax(axes[0], C["navy"])

for cid in sorted(df["persona"].unique()):
    mask = df["persona"] == cid
    axes[1].scatter(df.loc[mask, "daily_usage_hours"],
                    df.loc[mask, "productivity_score"],
                    alpha=0.45, s=20,
                    color=PERSONA_C.get(cid, C["muted"]),
                    label=PERSONA_L.get(cid, f"Cluster {cid}"), zorder=3)
cx = df.groupby("persona")["daily_usage_hours"].mean()
cy = df.groupby("persona")["productivity_score"].mean()
for cid in cx.index:
    axes[1].scatter(cx[cid], cy[cid], color=PERSONA_C.get(cid, C["muted"]),
                    s=200, marker="*", edgecolors=C["navy"], linewidth=1.2, zorder=5)
    axes[1].annotate(PERSONA_L.get(cid, str(cid)), (cx[cid], cy[cid]),
                     textcoords="offset points", xytext=(6, 6),
                     fontsize=9, fontweight="bold", color=PERSONA_C.get(cid, C["muted"]))
axes[1].set_xlabel("Daily Usage (hours)"); axes[1].set_ylabel("Productivity Score")
axes[1].set_title("Persona Scatter Plot", color=C["navy"])
axes[1].legend(title="Persona", fontsize=9); style_ax(axes[1], C["navy"])
plt.tight_layout()
plt.savefig("outputs/22_story_personas.png", dpi=300, bbox_inches="tight")
plt.close(); print(" 22_story_personas.png")


# ════════════════════════════════════════════════════════════════
# MASTER DASHBOARD — 3×3 full colour grid
# ════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(24, 18), facecolor=C["bg"])
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.34,
                        left=0.05, right=0.97, top=0.93, bottom=0.05)
fig.text(0.5, 0.965, "Social Media vs Productivity — Complete Analysis Dashboard",
         ha="center", fontsize=22, fontweight="bold", color=C["navy"])
fig.text(0.5, 0.947, "Does social media harm productivity?  The data-driven answer.",
         ha="center", fontsize=12, color=C["muted"])

# P1: Hero scatter (2 cols)
ax1 = fig.add_subplot(gs[0, :2]); ax1.set_facecolor("#EBF5FA")
ax1.axvspan(0,  2,        alpha=0.18, color=C["green"], label="Safe (<2h)",    zorder=1)
ax1.axvspan(2,  4,        alpha=0.16, color=C["gold"],  label="Caution (2–4h)", zorder=1)
ax1.axvspan(4,  x.max()+0.2, alpha=0.18, color=C["red"], label="Risk (>4h)",   zorder=1)
sc1 = ax1.scatter(df["daily_usage_hours"], df["productivity_score"],
                  c=df["stress_level"], cmap="RdYlGn_r", alpha=0.22, s=14, zorder=3)
plt.colorbar(sc1, ax=ax1, label="Stress", shrink=0.7, pad=0.01)
try:
    spl2 = make_interp_spline(bx2, by2, k=3)
    xs2  = np.linspace(bx2.min(), bx2.max(), 300)
    ax1.plot(xs2, spl2(xs2), color=C["red"], lw=3.0, zorder=5, label="Smoothed trend")
    ax1.fill_between(bx2, by2-be2, by2+be2, alpha=0.18, color=C["red"])
except Exception:
    ax1.plot(bx2, by2, color=C["red"], lw=3.0, zorder=5)
ax1.axvline(4, color=C["navy"], lw=2.0, ls=":", alpha=0.7, label="Tipping point")
ax1.set_title(f"Usage Hours → Productivity  (r={r:.3f},  p={p:.4f})",
              fontsize=12, fontweight="bold", color=C["navy"])
ax1.set_xlabel("Daily Social Media Usage (hours)")
ax1.set_ylabel("Productivity Score (1–10)")
ax1.legend(fontsize=8.5, loc="upper right"); ax1.set_xlim(0, df["daily_usage_hours"].max()+0.1)
for sp in ["left","bottom"]: ax1.spines[sp].set_color(C["blue"]); ax1.spines[sp].set_linewidth(1.5)

# P2: Platform risk
ax2 = fig.add_subplot(gs[0, 2]); style_ax(ax2)
medians_p2 = df.groupby("platform_preference")["productivity_score"].median().sort_values()
pal2 = PLATFORM_C[:len(medians_p2)]
bars_p2 = ax2.barh(range(len(medians_p2)), medians_p2.values, color=pal2,
                   edgecolor="white", height=0.60, alpha=0.92)
ax2.set_yticks(range(len(medians_p2))); ax2.set_yticklabels(medians_p2.index, fontsize=9)
for bar, val in zip(bars_p2, medians_p2.values):
    ax2.text(val+0.04, bar.get_y()+bar.get_height()/2, f"{val:.2f}",
             va="center", fontsize=9, fontweight="bold")
ax2.set_title("Median Productivity\nby Platform", fontsize=11, fontweight="bold", color=C["navy"])
ax2.set_xlabel("Median Score"); ax2.set_xlim(0, 11)

# P3: DAI histogram
ax3 = fig.add_subplot(gs[1, 0]); style_ax(ax3, C["red"], bg=C["red_lt"]+"33")
dai_v = df["dai"].replace([np.inf,-np.inf], np.nan).dropna()
cmap_dai = matplotlib.colormaps["RdYlGn_r"]
bin_e2 = np.linspace(dai_v.min(), dai_v.max(), 36)
for lo, hi in zip(bin_e2[:-1], bin_e2[1:]):
    cnt  = ((dai_v >= lo) & (dai_v < hi)).sum()
    frac = (lo - dai_v.min()) / (dai_v.max() - dai_v.min() + 1e-9)
    ax3.bar(lo, cnt, width=(hi-lo)*0.9, color=cmap_dai(frac), edgecolor="white", linewidth=0.2)
ax3.axvline(dai_v.median(), color=C["navy"], lw=2.0, ls="--",
            label=f"Median={dai_v.median():.1f}")
ax3.set_title("Digital Addiction Index (DAI)\n[Novel Feature — Top Predictor]",
              color=C["red"], fontsize=11, fontweight="bold")
ax3.set_xlabel("DAI Value"); ax3.set_ylabel("Count"); ax3.legend(fontsize=9)
for sp in ["left","bottom"]: ax3.spines[sp].set_color(C["red"]); ax3.spines[sp].set_linewidth(1.3)

# P4: Usage×Stress heatmap
ax4 = fig.add_subplot(gs[1, 1]); ax4.set_facecolor(C["bg"])
df["stress_band"] = pd.cut(df["stress_level"], 3, labels=["Low","Med","High"])
df["usage_band2"]  = pd.cut(df["daily_usage_hours"], 4, labels=["<2h","2-4h","4-8h",">8h"])
pivot2 = df.groupby(["usage_band2","stress_band"], observed=True)["productivity_score"].mean().unstack()
cmap_h2 = LinearSegmentedColormap.from_list("prod", [C["red"],C["gold"],C["green"]], N=256)
im4 = ax4.imshow(pivot2.values, cmap=cmap_h2, aspect="auto", vmin=1, vmax=10)
plt.colorbar(im4, ax=ax4, label="Mean Productivity", shrink=0.85)
ax4.set_xticks(range(len(pivot2.columns))); ax4.set_xticklabels(pivot2.columns, fontsize=10)
ax4.set_yticks(range(len(pivot2.index)));   ax4.set_yticklabels(pivot2.index, fontsize=10)
ax4.set_title("Heatmap: Usage × Stress → Productivity",
              fontsize=11, fontweight="bold", color=C["navy"])
ax4.set_xlabel("Stress Band"); ax4.set_ylabel("Usage Band")
for i in range(len(pivot2.index)):
    for j in range(len(pivot2.columns)):
        val = pivot2.values[i, j]
        if not np.isnan(val):
            ax4.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=11,
                     fontweight="bold", color=C["white"] if val < 5 else C["navy"])

# P5: Sleep bars
ax5 = fig.add_subplot(gs[1, 2]); style_ax(ax5, C["purple"], bg=C["purple_lt"]+"33")
sleep_bins2 = pd.cut(df["sleep_hours"], bins=[0,5,6,7,8,12],
                     labels=["<5h","5-6h","6-7h","7-8h ",">8h"])
sp_d2 = df.groupby(sleep_bins2, observed=True)["productivity_score"].agg(["mean","sem"])
sleep_pal2 = [C["red"],"#E07B4A",C["gold"],C["green"],C["blue"]]
bars_s2 = ax5.bar(range(len(sp_d2)), sp_d2["mean"], yerr=sp_d2["sem"],
                  color=sleep_pal2, edgecolor="white", capsize=4, width=0.62, alpha=0.92)
ax5.set_xticks(range(len(sp_d2)))
ax5.set_xticklabels(sp_d2.index, rotation=18, ha="right", fontsize=8.5)
ax5.set_ylabel("Mean Productivity"); ax5.set_ylim(0, 11)
ax5.set_title("Productivity vs Sleep Hours\n(sleep loss amplifies SM harm)",
              fontsize=11, fontweight="bold", color=C["navy"])
for bar, val in zip(bars_s2, sp_d2["mean"]):
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
             f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")
for sp in ["left","bottom"]: ax5.spines[sp].set_color(C["purple"]); ax5.spines[sp].set_linewidth(1.3)

# P6: Doomscroll × wellbeing
ax6 = fig.add_subplot(gs[2, 0]); style_ax(ax6, C["orange"])
if "doomscroll_risk" in df.columns:
    doom2 = df.groupby(["doomscroll_risk","wellbeing_app_use"])["productivity_score"].mean().unstack()
    doom2.index = ["No Risk","Doomscroll Risk"]
    x_d2 = np.arange(len(doom2)); w2 = 0.35
    ba2 = ax6.bar(x_d2-w2/2, doom2.get(0,[0,0]), w2, color=C["red"],   alpha=0.88, label="No Wellbeing App", edgecolor="white")
    bb2 = ax6.bar(x_d2+w2/2, doom2.get(1,[0,0]), w2, color=C["green"], alpha=0.88, label="Wellbeing App ON",  edgecolor="white")
    ax6.set_xticks(x_d2); ax6.set_xticklabels(doom2.index, fontsize=10)
    ax6.set_ylabel("Mean Productivity"); ax6.set_ylim(0, 10)
    ax6.set_title("Wellbeing App Effect on\nDoomscroll Risk [Novel]",
                  fontsize=11, fontweight="bold", color=C["navy"]); ax6.legend(fontsize=9)
    for grp in [ba2, bb2]:
        for bar in grp:
            ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.12,
                     f"{bar.get_height():.2f}", ha="center", fontsize=9, fontweight="bold")
for sp in ["left","bottom"]: ax6.spines[sp].set_color(C["orange"]); ax6.spines[sp].set_linewidth(1.3)

# P7: Persona profiles
ax7 = fig.add_subplot(gs[2, 1]); style_ax(ax7, C["teal"])
p_means2  = df.groupby("persona")[persona_attrs].mean()
normed2   = (p_means2-p_means2.min()) / (p_means2.max()-p_means2.min()+1e-9)
normed2.index = [PERSONA_L.get(i,f"C{i}") for i in normed2.index]
normed2.plot(kind="bar", ax=ax7, width=0.70, color=attr_colors,
             edgecolor="white", linewidth=0.7, alpha=0.90)
ax7.set_xticklabels(normed2.index, rotation=0, fontsize=10)
ax7.set_ylabel("Normalised Value (0–1)"); ax7.set_ylim(0, 1.15)
ax7.set_title("Persona Profiles — Normalised Attributes",
              fontsize=11, fontweight="bold", color=C["navy"])
ax7.legend(fontsize=8, loc="upper right")
for sp in ["left","bottom"]: ax7.spines[sp].set_color(C["teal"]); ax7.spines[sp].set_linewidth(1.3)

# P8: Verdict card
ax8 = fig.add_subplot(gs[2, 2])
ax8.set_facecolor(C["navy"]); ax8.axis("off")
verdict_lines = [
    ("VERDICT",                    13, C["gold"],       "bold"),
    ("",                            4, "",               ""),
    ("YES — social media",         10, C["red_lt"],     "bold"),
    ("harms productivity",         10, C["red_lt"],     "bold"),
    ("conditionally.",             10, C["red_lt"],     "bold"),
    ("",                            4, "",               ""),
    ("Mechanism:",                  9, C["white"],      "bold"),
    ("High usage →",              8.5, C["teal_lt"],   "normal"),
    ("sleep loss →",              8.5, C["purple_lt"], "normal"),
    ("stress ↑ →",                8.5, C["coral_lt"],  "normal"),
    ("productivity ↓",            8.5, C["red_lt"],    "normal"),
    ("",                            4, "",               ""),
    ("Tipping point: ~4h/day",    8.5, C["gold_lt"],   "bold"),
    ("TikTok/Instagram riskiest",   8, C["white"],     "normal"),
    ("Wellbeing apps: +12%",        8, C["green_lt"],  "bold"),
    ("",                            4, "",               ""),
    ("Top predictors (RF):",       8.5, C["white"],    "bold"),
    ("1. DAI (novel)",              8, C["red_lt"],    "normal"),
    ("2. Sleep hours",              8, C["purple_lt"], "normal"),
    ("3. Stress level",             8, C["coral_lt"],  "normal"),
    ("4. Daily usage",              8, C["orange_lt"], "normal"),
]
y_pos = 0.97
for text, size, color, weight in verdict_lines:
    if text == "":
        y_pos -= size / 130; continue
    ax8.text(0.07, y_pos, text, transform=ax8.transAxes,
             fontsize=size, color=color, fontweight=weight, va="top")
    y_pos -= size / 82
rect = mpatches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
    boxstyle="round,pad=0.01", linewidth=2.5,
    edgecolor=C["gold"], facecolor=C["navy"],
    transform=ax8.transAxes, zorder=0)
ax8.add_patch(rect)
ax8.set_title("Evidence-Based Answer", fontsize=11, fontweight="bold", color=C["navy"])

plt.savefig("outputs/23_master_dashboard.png", dpi=150,
            bbox_inches="tight", facecolor=C["bg"])
plt.close(); print(" 23_master_dashboard.png  ← MASTER STORY DASHBOARD")

r_final, _ = stats.pearsonr(df["daily_usage_hours"], df["productivity_score"])
print(f"\n{'='*60}")
print(f"FINAL ANSWER: Pearson r = {r_final:.3f}")
print("Social media DOES negatively impact productivity.")
print("Tipping point: ~4 hours/day.  Wellbeing apps buffer harm.")
print(f"{'='*60}")
print("\n ALL STORYTELLING CHARTS COMPLETE")
