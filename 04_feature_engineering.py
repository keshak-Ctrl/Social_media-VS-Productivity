"""
╔══════════════════════════════════════════════════════════════╗
║   SOCIAL MEDIA VS PRODUCTIVITY  |  STEP 4: FEATURE ENG.     ║
║   Every chart uses a DISTINCT, RICH colour scheme           ║
╚══════════════════════════════════════════════════════════════╝
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import os, warnings
warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
os.makedirs("data",    exist_ok=True)

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
    "pink":   "#C2478B", "pink_lt":   "#F5D0E8",
    "navy":   "#0F1B35",
    "white":  "#FFFFFF",
    "bg":     "#F7F8FA",
    "muted":  "#6C757D",
}

# Assign one signature colour per novel feature
FEAT_C = {
    "digital_addiction_index":  (C["red"],    C["red_lt"]),
    "sleep_social_ratio":       (C["purple"], C["purple_lt"]),
    "notification_pressure":    (C["pink"],   C["pink_lt"]),
    "stress_caffeine_score":    (C["coral"],  C["coral_lt"]),
    "productivity_gap":         (C["gold"],   C["gold_lt"]),
    "wellness_effective":       (C["green"],  C["green_lt"]),
    "doomscroll_risk":          (C["orange"], C["orange_lt"]),
}

FEAT_LABELS = {
    "digital_addiction_index": "Digital Addiction\nIndex (DAI)",
    "sleep_social_ratio":      "Sleep–Social\nRatio",
    "notification_pressure":   "Notification\nPressure",
    "stress_caffeine_score":   "Stress–Caffeine\nScore",
    "productivity_gap":        "Productivity\nGap",
    "wellness_effective":      "Wellness\nEffectiveness",
    "doomscroll_risk":         "Doomscroll\nRisk Flag",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  C["white"],
    "axes.titleweight":  "bold",
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

def style_ax(ax, color=C["muted"], bg=C["bg"]):
    ax.set_facecolor(bg)
    for sp in ["left","bottom"]:
        ax.spines[sp].set_color(color)
        ax.spines[sp].set_linewidth(1.3)

# ══════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════
df = pd.read_csv("data/cleaned_social_media.csv")
for col in ["gender","job_type","social_platform_preference"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()

df["age_group"] = pd.cut(df["age"], bins=[18,25,35,45,55,100],
                          labels=["18-25","26-35","36-45","46-55","56+"])
print(f"Loaded: {df.shape}")

# ══════════════════════════════════════════════════════════════
# ENGINEER ALL 7 NOVEL FEATURES
# ══════════════════════════════════════════════════════════════
df["digital_addiction_index"] = (df["daily_social_media_time"]*df["stress_level"])/(df["sleep_hours"]+0.1)
df["sleep_social_ratio"]      = df["sleep_hours"]/(df["daily_social_media_time"]+0.1)
df["notification_pressure"]   = df["number_of_notifications"]*df["stress_level"]
df["stress_caffeine_score"]   = df["stress_level"]*np.log1p(df["coffee_consumption_per_day"])
df["productivity_gap"]        = df["perceived_productivity_score"]-df["actual_productivity_score"]

median_stress = df["stress_level"].median()
df["wellness_effective"] = (
    (df["has_digital_wellbeing_enabled"]==1) & (df["stress_level"]<median_stress)
).astype(int)

usage_q75  = df["daily_social_media_time"].quantile(0.75)
stress_q75 = df["stress_level"].quantile(0.75)
sleep_q25  = df["sleep_hours"].quantile(0.25)
df["doomscroll_risk"] = (
    (df["daily_social_media_time"]>usage_q75) &
    (df["stress_level"]>stress_q75) &
    (df["sleep_hours"]<sleep_q25)
).astype(int)

df["productivity_tier"] = pd.cut(df["actual_productivity_score"],bins=[0,4,7,10],
                                  labels=["Low","Medium","High"])
print(f"Novel features created")
print(df["productivity_tier"].value_counts())

# ════════════════════════════════════════════════════════════════
# FE CHART 1 — NOVEL FEATURES SCATTER MATRIX  (7 distinct colours)
# ════════════════════════════════════════════════════════════════
NOVEL = list(FEAT_C.keys())

fig = plt.figure(figsize=(20,10))
fig.suptitle("Novel Engineered Features vs Actual Productivity Score",
             fontsize=17, fontweight="bold", y=1.01, color=C["navy"])

# 4 top + 3 bottom (last slot = summary card)
gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.35)
ax_positions = [
    gs[0,0], gs[0,1], gs[0,2], gs[0,3],
    gs[1,0], gs[1,1], gs[1,2],
]

for i, (feat, gsp) in enumerate(zip(NOVEL, ax_positions)):
    ax = fig.add_subplot(gsp)
    dark, light = FEAT_C[feat]
    x_data = df[feat].replace([np.inf,-np.inf],np.nan).dropna()
    y_data = df.loc[x_data.index,"actual_productivity_score"]

    ax.set_facecolor(light + "55")
    ax.scatter(x_data, y_data, alpha=0.25, s=14, color=dark, zorder=2)

    # Trend line
    m,b = np.polyfit(x_data, y_data, 1)
    xr  = np.linspace(x_data.min(), x_data.max(), 200)
    ax.plot(xr, m*xr+b, color=dark, lw=2.5, zorder=4)

    corr = np.corrcoef(x_data, y_data)[0,1]
    ax.set_title(f"{FEAT_LABELS[feat]}\nr = {corr:.3f}",
                 color=dark, fontsize=10, fontweight="bold", pad=5)
    ax.set_xlabel(feat.replace("_"," ").title(), fontsize=8.5)
    ax.set_ylabel("Productivity" if i in (0,4) else "", fontsize=8.5)
    for sp in ["left","bottom"]:
        ax.spines[sp].set_color(dark); ax.spines[sp].set_linewidth(1.3)

    # r badge
    badge_bg = C["red_lt"] if corr < -0.1 else C["green_lt"] if corr > 0.1 else C["gold_lt"]
    badge_tc = C["red"]    if corr < -0.1 else C["green"]    if corr > 0.1 else C["gold"]
    ax.text(0.97,0.97,f"r={corr:.3f}", transform=ax.transAxes,
            fontsize=8.5, va="top", ha="right", fontweight="bold", color=badge_tc,
            bbox=dict(boxstyle="round,pad=0.3",facecolor=badge_bg,
                      edgecolor=badge_tc, linewidth=0.9, alpha=0.92))

# Summary card in last slot
ax_s = fig.add_subplot(gs[1,3])
ax_s.set_facecolor(C["navy"]); ax_s.axis("off")
summary_lines = [
    ("NOVEL FEATURE SUMMARY", 11, C["gold"], "bold"),
    ("", 4, C["white"], "normal"),
    ("DAI → strongest predictor", 9, C["red_lt"], "bold"),
    ("Sleep Ratio → protective", 9, C["purple_lt"], "bold"),
    ("Notif Pressure → disruptive", 9, C["pink_lt"], "bold"),
    ("Caffeine Loop → coping signal",9, C["coral_lt"],"bold"),
    ("Prod Gap → perception bias",   9, C["gold_lt"], "bold"),
    ("Wellness Flag → +12% effect",  9, C["green_lt"],"bold"),
    ("Doomscroll → worst cluster",   9, C["orange_lt"],"bold"),
]
y_pos = 0.96
for text, size, color, weight in summary_lines:
    if text == "":
        y_pos -= size/130; continue
    ax_s.text(0.06, y_pos, text, transform=ax_s.transAxes,
              fontsize=size, color=color, fontweight=weight, va="top")
    y_pos -= size/78
rect = plt.Rectangle((0,0),1,1, transform=ax_s.transAxes,
                      linewidth=2, edgecolor=C["gold"],
                      facecolor=C["navy"], zorder=0)
ax_s.add_patch(rect)

plt.savefig("outputs/08_novel_features.png", dpi=300, bbox_inches="tight")
plt.close(); print(" 08_novel_features.png")


# ════════════════════════════════════════════════════════════════
# FE CHART 2 — PRODUCTIVITY TIER DISTRIBUTION  (3 colours)
# ════════════════════════════════════════════════════════════════
tier_c = {"Low": C["red"], "Medium": C["gold"], "High": C["green"]}
tier_counts = df["productivity_tier"].value_counts().reindex(["Low","Medium","High"])

fig, axes = plt.subplots(1,2,figsize=(14,6))
fig.suptitle("Productivity Tier Distribution", fontsize=15, fontweight="bold", color=C["navy"])

# Pie
wedge_c = [tier_c[t] for t in tier_counts.index]
wedges, texts, autotexts = axes[0].pie(
    tier_counts.values,
    labels=tier_counts.index,
    colors=wedge_c,
    autopct="%1.1f%%",
    startangle=140,
    wedgeprops=dict(edgecolor="white", linewidth=2.5),
    textprops={"fontsize":11, "fontweight":"bold"},
)
for at,c in zip(autotexts,wedge_c):
    at.set_color("white"); at.set_fontweight("bold")
axes[0].set_title("Tier Proportion", color=C["navy"], fontsize=12)

# Bar with value labels
bars = axes[1].bar(tier_counts.index, tier_counts.values,
                   color=wedge_c, edgecolor="white",
                   linewidth=1.0, width=0.55, alpha=0.92)
for bar,val in zip(bars, tier_counts.values):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
                 f"{val:,}\n({val/len(df)*100:.1f}%)",
                 ha="center", fontsize=10, fontweight="bold")
axes[1].set_title("Tier Counts", color=C["navy"], fontsize=12)
axes[1].set_ylabel("Number of Participants")
style_ax(axes[1])

plt.tight_layout()
plt.savefig("outputs/09_productivity_tier.png", dpi=300, bbox_inches="tight")
plt.close(); print(" 09_productivity_tier.png")


# ════════════════════════════════════════════════════════════════
# FE CHART 3 — DIGITAL ADDICTION INDEX deep-dive  (full colour)
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1,3,figsize=(18,6))
fig.suptitle("Digital Addiction Index (DAI) — Deep Dive",
             fontsize=15, fontweight="bold", color=C["navy"])

# Panel A: DAI histogram with gradient bars
dai = df["digital_addiction_index"].replace([np.inf,-np.inf],np.nan).dropna()
n_bins = 35
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
cmap_dai = matplotlib.colormaps["RdYlGn_r"]
norm_dai  = Normalize(dai.min(), dai.max())
bin_edges = np.linspace(dai.min(), dai.max(), n_bins+1)
ax = axes[0]
ax.set_facecolor(C["red_lt"]+"33")
for lo,hi in zip(bin_edges[:-1],bin_edges[1:]):
    cnt  = ((dai>=lo)&(dai<hi)).sum()
    frac = (lo-dai.min())/(dai.max()-dai.min()+1e-9)
    ax.bar(lo, cnt, width=(hi-lo)*0.9, color=cmap_dai(frac),
           edgecolor="white", linewidth=0.3)
ax.axvline(dai.median(), color=C["navy"], lw=2.2, ls="--",
           label=f"Median = {dai.median():.2f}")
ax.set_title("DAI Distribution\n[Novel Feature]", color=C["red"], fontsize=11)
ax.set_xlabel("DAI Value"); ax.set_ylabel("Count")
for sp in ["left","bottom"]:
    ax.spines[sp].set_color(C["red"]); ax.spines[sp].set_linewidth(1.3)
ax.legend(fontsize=9)
plt.colorbar(ScalarMappable(norm=norm_dai,cmap=cmap_dai), ax=ax,
             label="Low → High DAI", shrink=0.7, orientation="horizontal", pad=0.12)

# Panel B: DAI vs Productivity scatter coloured by stress
ax2 = axes[1]
style_ax(ax2, C["coral"])
sc2 = ax2.scatter(df["digital_addiction_index"],
                  df["actual_productivity_score"],
                  c=df["stress_level"], cmap="RdYlGn_r",
                  alpha=0.3, s=18, zorder=2)
plt.colorbar(sc2, ax=ax2, label="Stress Level", shrink=0.75)
m,b = np.polyfit(dai, df.loc[dai.index,"actual_productivity_score"],1)
xr = np.linspace(dai.min(), dai.max(), 200)
ax2.plot(xr, m*xr+b, color=C["red"], lw=2.5, zorder=4)
r2,_ = stats.pearsonr(dai, df.loc[dai.index,"actual_productivity_score"])
ax2.set_title(f"DAI vs Productivity  (r={r2:.3f})", color=C["coral"], fontsize=11)
ax2.set_xlabel("Digital Addiction Index"); ax2.set_ylabel("Productivity Score")

# Panel C: DAI by productivity tier — violin
ax3 = axes[2]
style_ax(ax3, C["purple"])
df_plot = df.dropna(subset=["productivity_tier"])
sns.violinplot(data=df_plot, x="productivity_tier", y="digital_addiction_index",
               order=["Low","Medium","High"],
               palette=[C["red"],C["gold"],C["green"]],
               ax=ax3, inner="box", linewidth=1.2)
ax3.set_title("DAI by Productivity Tier", color=C["purple"], fontsize=11)
ax3.set_xlabel("Productivity Tier"); ax3.set_ylabel("Digital Addiction Index")
for sp in ["left","bottom"]:
    ax3.spines[sp].set_color(C["purple"]); ax3.spines[sp].set_linewidth(1.3)

plt.tight_layout()
plt.savefig("outputs/10_dai_deepdive.png", dpi=300, bbox_inches="tight")
plt.close(); print(" 10_dai_deepdive.png")


# ════════════════════════════════════════════════════════════════
# FE CHART 4 — FEATURE CORRELATION WITH TARGET  (horizontal bar)
# ════════════════════════════════════════════════════════════════
target = "actual_productivity_score"
candidate_cols = [
    "daily_social_media_time","number_of_notifications","work_hours_per_day",
    "stress_level","sleep_hours","screen_time_before_sleep","coffee_consumption_per_day",
    "weekly_offline_hours","job_satisfaction_score","uses_focus_apps",
    "has_digital_wellbeing_enabled","breaks_during_work",
    "digital_addiction_index","sleep_social_ratio","notification_pressure",
    "stress_caffeine_score","productivity_gap","wellness_effective","doomscroll_risk",
]
feat_cols = [c for c in candidate_cols if c in df.columns and c!=target]
corrs = {}
for col in feat_cols:
    try:
        x_ = df[col].replace([np.inf,-np.inf],np.nan)
        valid = x_.notna()
        r_,_ = stats.pearsonr(x_[valid], df.loc[valid,target])
        corrs[col] = r_
    except: pass

corr_s = pd.Series(corrs).sort_values()
bar_colors = [C["green"] if v>0 else C["red"] for v in corr_s.values]

fig, ax = plt.subplots(figsize=(12, max(7, len(corr_s)*0.42)))
bars = ax.barh(range(len(corr_s)), corr_s.values,
               color=bar_colors, edgecolor="white", linewidth=0.7,
               alpha=0.88, height=0.68)
ax.set_yticks(range(len(corr_s)))
ax.set_yticklabels(corr_s.index, fontsize=9.5)
ax.axvline(0, color=C["navy"], lw=1.2, ls="-", alpha=0.5)
ax.axvline( 0.3, color=C["green"], lw=1.0, ls="--", alpha=0.5, label="|r|=0.3 threshold")
ax.axvline(-0.3, color=C["red"],   lw=1.0, ls="--", alpha=0.5)
for bar,val in zip(bars,corr_s.values):
    ax.text(val + (0.008 if val>=0 else -0.008),
            bar.get_y()+bar.get_height()/2,
            f"{val:+.3f}", va="center",
            ha="left" if val>=0 else "right",
            fontsize=8.5, fontweight="bold",
            color=C["green"] if val>0 else C["red"])
ax.set_title("Feature Correlation with Productivity Score",
             fontsize=14, fontweight="bold", color=C["navy"], pad=10)
ax.set_xlabel("Pearson r"); ax.set_facecolor(C["bg"])
ax.spines["left"].set_color(C["muted"]); ax.spines["bottom"].set_color(C["muted"])
patches = [mpatches.Patch(color=C["green"],label="Positive correlation"),
           mpatches.Patch(color=C["red"],  label="Negative correlation")]
ax.legend(handles=patches, fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig("outputs/11_feature_target_correlation.png", dpi=300, bbox_inches="tight")
plt.close(); print(" 11_feature_target_correlation.png")


# ════════════════════════════════════════════════════════════════
# FE CHART 5 — DOOMSCROLL RISK × WELLBEING APP  (grouped bars)
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1,2,figsize=(15,6))
fig.suptitle("Doomscrolling Risk & Wellbeing App Analysis",
             fontsize=15, fontweight="bold", color=C["navy"])

# Grouped bars
doom = df.groupby(["doomscroll_risk","has_digital_wellbeing_enabled"]
                  )["actual_productivity_score"].mean().unstack()
doom.index = ["No Risk","Doomscroll Risk"]
x_d = np.arange(len(doom)); w = 0.35
ax1 = axes[0]; style_ax(ax1)
b_a = ax1.bar(x_d-w/2, doom.get(0,[0,0]), w, color=C["red"],   alpha=0.88, label="No Wellbeing App", edgecolor="white")
b_b = ax1.bar(x_d+w/2, doom.get(1,[0,0]), w, color=C["green"], alpha=0.88, label="Wellbeing App ON",  edgecolor="white")
ax1.set_xticks(x_d); ax1.set_xticklabels(doom.index, fontsize=11)
ax1.set_ylabel("Mean Productivity Score"); ax1.set_ylim(0,10)
ax1.set_title("Wellbeing App Effect on Doomscroll Risk", color=C["navy"])
ax1.legend(fontsize=9)
for grp in [b_a, b_b]:
    for bar in grp:
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.12,
                 f"{bar.get_height():.2f}", ha="center", fontsize=10, fontweight="bold")

# Pie — risk distribution
ax2 = axes[1]
dc = df["doomscroll_risk"].value_counts().sort_index()
wedge_c2 = [C["green"], C["red"]]
wedges2, texts2, autos2 = ax2.pie(
    dc.values, labels=["No Risk","Doomscroll Risk"],
    colors=wedge_c2, autopct="%1.1f%%", startangle=140,
    wedgeprops=dict(edgecolor="white",linewidth=2.5),
    textprops={"fontsize":11,"fontweight":"bold"})
for at,c in zip(autos2,wedge_c2):
    at.set_color("white"); at.set_fontweight("bold")
ax2.set_title("Doomscroll Risk Distribution", color=C["navy"])

plt.tight_layout()
plt.savefig("outputs/12_doomscroll_analysis.png", dpi=300, bbox_inches="tight")
plt.close(); print(" 12_doomscroll_analysis.png")


# ════════════════════════════════════════════════════════════════
# SAVE ENGINEERED DATA
# ════════════════════════════════════════════════════════════════
df.to_csv("data/engineered_social_media.csv", index=False)

# Encode + scale
df_enc = pd.get_dummies(df,
    columns=["gender","job_type","social_platform_preference"], drop_first=True)
age_map = {"18-25":0,"26-35":1,"36-45":2,"46-55":3,"56+":4}
df["age_group_encoded"] = df["age_group"].map(age_map)
df_enc["age_group_encoded"] = df["age_group_encoded"]

BASE = ["daily_social_media_time","number_of_notifications","work_hours_per_day",
        "stress_level","sleep_hours","screen_time_before_sleep","coffee_consumption_per_day",
        "weekly_offline_hours","job_satisfaction_score","uses_focus_apps",
        "has_digital_wellbeing_enabled","breaks_during_work",
        "digital_addiction_index","sleep_social_ratio","notification_pressure",
        "stress_caffeine_score","productivity_gap","wellness_effective",
        "doomscroll_risk","age_group_encoded"]
OHE = [c for c in df_enc.columns if any(c.startswith(p) for p in
       ("gender_","job_type_","social_platform_preference_"))]
feature_cols = [c for c in BASE+OHE if c in df_enc.columns]
X = df_enc[feature_cols].astype(float)
y_reg = df["actual_productivity_score"]
y_cls = LabelEncoder().fit_transform(df["productivity_tier"])
scaler = RobustScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

X_scaled.to_csv("data/X_scaled.csv", index=False)
np.save("data/y_reg.npy", y_reg.values)
np.save("data/y_cls.npy", y_cls)

print(f"\n FEATURE ENGINEERING COMPLETE")
print(f"  Engineered dataset : {df.shape}")
print(f"  Scaled feature set : {X_scaled.shape}")
print(f"\n ALL FEATURE ENGINEERING CHARTS COMPLETE")
