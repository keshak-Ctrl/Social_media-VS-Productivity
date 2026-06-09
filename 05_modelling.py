"""
╔══════════════════════════════════════════════════════════════╗
║   SOCIAL MEDIA VS PRODUCTIVITY  |  STEP 5: ML PIPELINE      ║
║   Every chart uses a DISTINCT, RICH colour scheme           ║
╚══════════════════════════════════════════════════════════════╝
Regression · Classification · Clustering · Feature Importance
"""
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import label_binarize
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models",  exist_ok=True)

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
TIER_C   = {"Low": C["red"],   "Medium": C["gold"],   "High": C["green"]}
PERSONA_C = {0: C["green"], 1: C["red"], 2: C["orange"]}
PERSONA_L = {0: "Balanced",    1: "At-Risk",           2: "Moderate"}

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
    "legend.fontsize":   9,
    "legend.framealpha": 0.9,
})

def style_ax(ax, color=C["muted"], bg=C["bg"]):
    ax.set_facecolor(bg)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(color)
        ax.spines[sp].set_linewidth(1.3)

def metric_card(ax, title, value, color, sub=""):
    ax.set_facecolor(color + "22")
    ax.axis("off")
    ax.text(0.5, 0.62, value,  transform=ax.transAxes, ha="center",
            fontsize=28, fontweight="bold", color=color)
    ax.text(0.5, 0.28, title,  transform=ax.transAxes, ha="center",
            fontsize=9,  fontweight="bold", color=C["navy"])
    if sub:
        ax.text(0.5, 0.10, sub, transform=ax.transAxes, ha="center",
                fontsize=8, color=C["muted"])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(color)
        sp.set_linewidth(2)

# ══════════════════════════════════════════════════════════════
# LOAD & PREPARE
# ══════════════════════════════════════════════════════════════
df = pd.read_csv("data/engineered_social_media.csv")
print(f"Dataset: {df.shape}")

for col in ["gender","job_type","social_platform_preference"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()

df_model = pd.get_dummies(df,
    columns=["gender","job_type","social_platform_preference"],
    drop_first=True)

# Targets
y_reg = df_model["actual_productivity_score"].copy()

if "productivity_tier" not in df_model.columns:
    df_model["productivity_tier"] = pd.cut(
        df_model["actual_productivity_score"],
        bins=[0,4,7,10], labels=["Low","Medium","High"])

tier_map = {"Low":0,"Medium":1,"High":2}
y_cls = df_model["productivity_tier"].map(tier_map)

# Features
DROP = ["actual_productivity_score","productivity_tier",
        "age_group","usage_band","stress_band","productivity_gap",
        "perceived_productivity_score","persona_cluster"]
DROP = [c for c in DROP if c in df_model.columns]
feature_cols = [c for c in df_model.columns if c not in DROP]

X = df_model[feature_cols].copy()
X.replace([np.inf,-np.inf], np.nan, inplace=True)
X = X.astype({c:int for c in X.columns if X[c].dtype==bool})
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
print(f"Features: {X.shape[1]}  NaNs: {X.isnull().sum().sum()}")

# ══════════════════════════════════════════════════════════════
# TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.20, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_cls, test_size=0.20, random_state=42, stratify=y_cls)

# ══════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════
ridge  = Ridge(alpha=1.0)
rf_reg = RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=-1)
rf_clf = RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1,class_weight="balanced")

ridge.fit(X_train, y_train)
rf_reg.fit(X_train, y_train)
rf_clf.fit(X_train_c, y_train_c)

ridge_pred = ridge.predict(X_test)
rf_pred    = rf_reg.predict(X_test)
y_pred_c   = rf_clf.predict(X_test_c)
cv_scores  = cross_val_score(rf_reg, X, y_reg, cv=5, scoring="r2")
cv_acc     = cross_val_score(rf_clf, X, y_cls, cv=5, scoring="accuracy")

ridge_r2 = r2_score(y_test, ridge_pred)
rf_r2    = r2_score(y_test, rf_pred)
rf_mae   = mean_absolute_error(y_test, rf_pred)
rf_rmse  = mean_squared_error(y_test, rf_pred)**0.5
clf_acc  = accuracy_score(y_test_c, y_pred_c)

print(f"\nRidge R²={ridge_r2:.4f}  RF R²={rf_r2:.4f}  MAE={rf_mae:.4f}  RMSE={rf_rmse:.4f}")
print(f"Clf Acc={clf_acc:.4f}  CV R²={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

# ══════════════════════════════════════════════════════════════
# ML CHART 1 — MODEL METRICS DASHBOARD  (metric cards + scatter)
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20,12))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38,
                        left=0.06, right=0.97, top=0.92, bottom=0.06)
fig.suptitle("Machine Learning Model Performance Dashboard",
             fontsize=18, fontweight="bold", color=C["navy"], y=0.97)

# ── Row 0: KPI cards ──────────────────────────────────────────
for col_i,(title,val,color,sub) in enumerate([
    ("RF  R²",          f"{rf_r2:.4f}",    C["teal"],  "Regression fit"),
    ("Ridge  R²",       f"{ridge_r2:.4f}", C["blue"],  "Baseline"),
    ("MAE",             f"{rf_mae:.4f}",   C["gold"],  "Mean Abs Error"),
    ("Classifier Acc.", f"{clf_acc*100:.1f}%", C["purple"],"RF tier accuracy"),
]):
    ax = fig.add_subplot(gs[0, col_i])
    metric_card(ax, title, val, color, sub)

# ── Row 1 col 0-1: Actual vs Predicted scatter ────────────────
ax1 = fig.add_subplot(gs[1, :2])
style_ax(ax1, C["teal"])
ax1.scatter(y_test, rf_pred, c=rf_pred, cmap="RdYlGn",
            alpha=0.45, s=22, zorder=3)
lims = [min(y_test.min(), rf_pred.min())-0.3,
        max(y_test.max(), rf_pred.max())+0.3]
ax1.plot(lims, lims, color=C["red"], lw=2.2, ls="--", label="Perfect fit", zorder=4)
ax1.fill_between(lims, [l-1 for l in lims], [l+1 for l in lims],
                 alpha=0.08, color=C["teal"], label="±1 pt band")
ax1.set_xlim(lims); ax1.set_ylim(lims)
ax1.set_xlabel("Actual Productivity Score"); ax1.set_ylabel("Predicted Score")
ax1.set_title(f"RF Regression — Actual vs Predicted  (R²={rf_r2:.4f}, RMSE={rf_rmse:.4f})",
              color=C["navy"])
ax1.legend(fontsize=9)

# ── Row 1 col 2-3: Residuals distribution ─────────────────────
ax2 = fig.add_subplot(gs[1, 2:])
style_ax(ax2, C["coral"])
residuals = y_test.values - rf_pred
ax2.set_facecolor(C["coral_lt"]+"33")
sns.histplot(residuals, kde=True, ax=ax2,
             color=C["coral"], alpha=0.72, edgecolor="white", linewidth=0.5,
             line_kws={"color":C["coral"],"lw":2.5})
ax2.axvline(0,       color=C["navy"],  lw=2.2, ls="--", label="Zero error")
ax2.axvline( rf_rmse,color=C["red"],   lw=1.8, ls=":",  label=f"+RMSE ({rf_rmse:.3f})")
ax2.axvline(-rf_rmse,color=C["red"],   lw=1.8, ls=":",  label=f"−RMSE")
ax2.axvspan(-rf_rmse, rf_rmse, alpha=0.10, color=C["coral"])
ax2.set_xlabel("Residual (Actual − Predicted)"); ax2.set_ylabel("Count")
ax2.set_title("Residual Distribution — RF Regression", color=C["coral"])
ax2.legend(fontsize=8.5)
for sp in ["left","bottom"]:
    ax2.spines[sp].set_color(C["coral"]); ax2.spines[sp].set_linewidth(1.3)

# ── Row 2 col 0-1: Ridge vs RF scatter overlay ────────────────
ax3 = fig.add_subplot(gs[2, :2])
style_ax(ax3, C["blue"])
ax3.scatter(y_test, rf_pred,    alpha=0.35, s=18, color=C["teal"],  label=f"Random Forest (R²={rf_r2:.3f})",  zorder=3)
ax3.scatter(y_test, ridge_pred, alpha=0.35, s=18, color=C["orange"],label=f"Ridge Reg.    (R²={ridge_r2:.3f})", zorder=3)
ax3.plot(lims, lims, color=C["navy"], lw=2.0, ls="--", label="Perfect fit", zorder=4)
ax3.set_xlim(lims); ax3.set_ylim(lims)
ax3.set_xlabel("Actual Productivity Score"); ax3.set_ylabel("Predicted Score")
ax3.set_title("Model Comparison — RF vs Ridge Regression", color=C["navy"])
ax3.legend(fontsize=9)

# ── Row 2 col 2-3: CV score distribution ─────────────────────
ax4 = fig.add_subplot(gs[2, 2:])
style_ax(ax4, C["purple"])
ax4.set_facecolor(C["purple_lt"]+"33")
x_cv = range(1, len(cv_scores)+1)
ax4.bar(x_cv, cv_scores, color=C["purple"], alpha=0.78,
        edgecolor="white", linewidth=0.8, width=0.55)
ax4.axhline(cv_scores.mean(), color=C["red"], lw=2.2, ls="--",
            label=f"Mean CV R² = {cv_scores.mean():.4f}")
ax4.axhspan(cv_scores.mean()-cv_scores.std(),
            cv_scores.mean()+cv_scores.std(),
            alpha=0.12, color=C["red"], label=f"±1σ = {cv_scores.std():.4f}")
for i,v in enumerate(cv_scores):
    ax4.text(i+1, v+0.002, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
ax4.set_xlabel("Fold"); ax4.set_ylabel("R² Score")
ax4.set_title("5-Fold Cross-Validation R² Scores", color=C["purple"])
ax4.legend(fontsize=8.5)
for sp in ["left","bottom"]:
    ax4.spines[sp].set_color(C["purple"]); ax4.spines[sp].set_linewidth(1.3)

plt.savefig("outputs/13_model_performance.png", dpi=300, bbox_inches="tight")
plt.close(); print(" 13_model_performance.png")


# ══════════════════════════════════════════════════════════════
# ML CHART 2 — FEATURE IMPORTANCE  (top 20, colour-coded)
# ══════════════════════════════════════════════════════════════
importance = (pd.DataFrame({"Feature":feature_cols,
                             "Importance":rf_reg.feature_importances_})
              .sort_values("Importance",ascending=False).head(20))

# Tag features by category for colour coding
def feat_color(name):
    if any(k in name for k in ["digital_addiction","sleep_social","notification_pressure",
                                 "stress_caffeine","productivity_gap","wellness","doomscroll"]):
        return C["purple"], "Novel Feature"
    if any(k in name for k in ["sleep","stress","coffee","screen_time","offline"]):
        return C["coral"],  "Wellbeing"
    if any(k in name for k in ["social_media","notification","platform","focus_app","wellbeing_enabled"]):
        return C["red"],    "Social Media"
    if any(k in name for k in ["job","work","satisfaction","breaks"]):
        return C["blue"],   "Work"
    return C["teal"], "Demographic"

colors_fi = [feat_color(f)[0] for f in importance["Feature"]]
labels_fi  = [feat_color(f)[1] for f in importance["Feature"]]

fig, ax = plt.subplots(figsize=(13,10))
style_ax(ax, C["muted"])
bars = ax.barh(range(len(importance)), importance["Importance"].values,
               color=colors_fi, edgecolor="white", linewidth=0.8,
               alpha=0.90, height=0.70)
ax.set_yticks(range(len(importance)))
ax.set_yticklabels(importance["Feature"], fontsize=9.5)
ax.invert_yaxis()
for bar, val in zip(bars, importance["Importance"].values):
    ax.text(val+0.0005, bar.get_y()+bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8.5, fontweight="bold")
ax.set_xlabel("Feature Importance (Mean Decrease Impurity)")
ax.set_title("Top 20 Feature Importances — Random Forest Regressor",
             fontsize=14, fontweight="bold", color=C["navy"], pad=12)
legend_items = {
    "Novel Feature": C["purple"],
    "Wellbeing":     C["coral"],
    "Social Media":  C["red"],
    "Work":          C["blue"],
    "Demographic":   C["teal"],
}
patches = [mpatches.Patch(color=c, label=l) for l,c in legend_items.items()]
ax.legend(handles=patches, fontsize=9, loc="lower right",
          title="Feature Category", title_fontsize=9)
plt.tight_layout()
plt.savefig("outputs/14_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close(); print(" 14_feature_importance.png")


# ══════════════════════════════════════════════════════════════
# ML CHART 3 — CLASSIFICATION RESULTS  (confusion + per-class)
# ══════════════════════════════════════════════════════════════
cm      = confusion_matrix(y_test_c, y_pred_c)
labels  = ["Low","Medium","High"]
cm_norm = cm.astype(float) / cm.sum(axis=1)[:,np.newaxis]

fig, axes = plt.subplots(1, 3, figsize=(19, 6))
fig.suptitle("Classification Results — Productivity Tier Prediction",
             fontsize=16, fontweight="bold", color=C["navy"])

# Panel A: Raw confusion matrix
ax = axes[0]
cmap_cm = LinearSegmentedColormap.from_list("cm", [C["white"], C["purple"]])
im = ax.imshow(cm, cmap=cmap_cm, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.80, label="Count")
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=11)
ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("Predicted Tier", fontsize=11); ax.set_ylabel("Actual Tier", fontsize=11)
ax.set_title(f"Confusion Matrix  (Acc={clf_acc:.3f})", color=C["navy"])
thresh = cm.max() / 2
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cm[i,j]}", ha="center", va="center", fontsize=13,
                fontweight="bold",
                color=C["white"] if cm[i,j]>thresh else C["navy"])

# Panel B: Normalised confusion (%)
ax2 = axes[1]
cmap_n = LinearSegmentedColormap.from_list("cmn", [C["white"], C["teal"]])
im2 = ax2.imshow(cm_norm, cmap=cmap_n, aspect="auto", vmin=0, vmax=1)
plt.colorbar(im2, ax=ax2, shrink=0.80, label="Recall")
ax2.set_xticks(range(3)); ax2.set_xticklabels(labels, fontsize=11)
ax2.set_yticks(range(3)); ax2.set_yticklabels(labels, fontsize=11)
ax2.set_xlabel("Predicted Tier", fontsize=11); ax2.set_ylabel("Actual Tier", fontsize=11)
ax2.set_title("Normalised Confusion Matrix (Recall %)", color=C["navy"])
for i in range(3):
    for j in range(3):
        ax2.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                 fontsize=12, fontweight="bold",
                 color=C["white"] if cm_norm[i,j]>0.5 else C["navy"])

# Panel C: Per-class metrics bar chart
report = classification_report(y_test_c, y_pred_c,
                                target_names=labels, output_dict=True)
metrics_names = ["precision","recall","f1-score"]
x_pos = np.arange(len(labels))
w = 0.25
ax3 = axes[2]
style_ax(ax3, C["blue"])
for i,(metric,color) in enumerate(zip(metrics_names,[C["blue"],C["orange"],C["purple"]])):
    vals = [report[t][metric] for t in labels]
    bars3 = ax3.bar(x_pos + i*w, vals, width=w, color=color,
                    edgecolor="white", linewidth=0.7, alpha=0.88, label=metric.title())
    for bar,val in zip(bars3,vals):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                 f"{val:.2f}", ha="center", fontsize=8, fontweight="bold")
ax3.set_xticks(x_pos + w); ax3.set_xticklabels(labels, fontsize=11)
ax3.set_ylabel("Score"); ax3.set_ylim(0, 1.12)
ax3.set_title("Per-Class Precision / Recall / F1", color=C["navy"])
ax3.legend(fontsize=9)
ax3.axhline(0.8, color=C["muted"], lw=1.2, ls="--", alpha=0.6, label="0.8 threshold")

plt.tight_layout()
plt.savefig("outputs/15_classification_results.png", dpi=300, bbox_inches="tight")
plt.close(); print(" 15_classification_results.png")


# ══════════════════════════════════════════════════════════════
# ML CHART 4 — LEARNING CURVES  (train vs val, 2 models)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Learning Curves — Training vs Validation",
             fontsize=15, fontweight="bold", color=C["navy"])

for ax, model, title, color in [
    (axes[0], rf_reg,  "Random Forest Regressor",  C["teal"]),
    (axes[1], rf_clf,  "Random Forest Classifier",  C["purple"]),
]:
    scoring = "r2" if "Regressor" in title else "accuracy"
    target  = y_reg if "Regressor" in title else y_cls
    sizes, train_sc, val_sc = learning_curve(
        model, X, target, cv=5, scoring=scoring,
        train_sizes=np.linspace(0.1,1.0,8), n_jobs=-1)

    t_mean, t_std = train_sc.mean(1), train_sc.std(1)
    v_mean, v_std = val_sc.mean(1),   val_sc.std(1)

    style_ax(ax, color)
    ax.plot(sizes, t_mean, "o-", color=color,     lw=2.2, label="Training score")
    ax.plot(sizes, v_mean, "s-", color=C["orange"],lw=2.2, label="Validation score")
    ax.fill_between(sizes, t_mean-t_std, t_mean+t_std, alpha=0.15, color=color)
    ax.fill_between(sizes, v_mean-v_std, v_mean+v_std, alpha=0.15, color=C["orange"])
    ax.set_title(title, color=color, fontsize=12)
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel(scoring.upper().replace("_"," "))
    ax.legend(fontsize=9)
    ax.set_facecolor(color+"11")
    for sp in ["left","bottom"]:
        ax.spines[sp].set_color(color); ax.spines[sp].set_linewidth(1.3)

plt.tight_layout()
plt.savefig("outputs/16_learning_curves.png", dpi=300, bbox_inches="tight")
plt.close(); print(" 16_learning_curves.png")


# ══════════════════════════════════════════════════════════════
# ML CHART 5 — CLUSTERING / USER PERSONAS  (rich scatter)
# ══════════════════════════════════════════════════════════════
CLUSTER_COLS = [c for c in ["daily_social_media_time","sleep_hours","stress_level",
                              "digital_addiction_index","sleep_social_ratio",
                              "notification_pressure"] if c in df.columns]
cluster_df = df[CLUSTER_COLS].replace([np.inf,-np.inf],np.nan).fillna(
    df[CLUSTER_COLS].median())
kmeans   = KMeans(n_clusters=3, random_state=42, n_init=20)
clusters = kmeans.fit_predict(cluster_df)
df["persona_cluster"] = clusters
sil = silhouette_score(cluster_df, clusters)
print(f"Silhouette: {sil:.4f}")

fig = plt.figure(figsize=(20, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
fig.suptitle(f"User Persona Clusters — K-Means (k=3, Silhouette={sil:.4f})",
             fontsize=17, fontweight="bold", color=C["navy"], y=0.99)

# Main scatter — usage vs productivity coloured by persona
ax_main = fig.add_subplot(gs[:, 0])
style_ax(ax_main, C["navy"])
for cid in sorted(df["persona_cluster"].unique()):
    mask = df["persona_cluster"]==cid
    ax_main.scatter(df.loc[mask,"daily_social_media_time"],
                    df.loc[mask,"actual_productivity_score"],
                    alpha=0.50, s=22, color=PERSONA_C[cid],
                    label=PERSONA_L[cid], zorder=3)
    cx = df.loc[mask,"daily_social_media_time"].mean()
    cy = df.loc[mask,"actual_productivity_score"].mean()
    ax_main.scatter(cx, cy, color=PERSONA_C[cid], s=180, marker="*",
                    edgecolors=C["navy"], linewidth=1.2, zorder=5)
    ax_main.annotate(PERSONA_L[cid], (cx,cy),
                     textcoords="offset points", xytext=(6,6),
                     fontsize=9, fontweight="bold", color=PERSONA_C[cid])
ax_main.set_xlabel("Daily Social Media Usage (hours)")
ax_main.set_ylabel("Actual Productivity Score")
ax_main.set_title("Usage vs Productivity\nby Persona Cluster", color=C["navy"])
ax_main.legend(title="Persona", fontsize=9, title_fontsize=9)

# Radar-style: persona profile bars per attribute (4 panels)
profile_attrs = [
    ("actual_productivity_score", "Productivity",  C["teal"]),
    ("daily_social_media_time",   "SM Usage",      C["red"]),
    ("sleep_hours",               "Sleep",         C["purple"]),
    ("stress_level",              "Stress",        C["coral"]),
    ("digital_addiction_index",   "DAI",           C["orange"]),
]
positions = [gs[0,1], gs[0,2], gs[1,1], gs[1,2]]
for i, (attr, label, attr_c) in enumerate(profile_attrs[:4]):
    ax_p = fig.add_subplot(positions[i])
    style_ax(ax_p, attr_c, bg=attr_c+"11")
    means = [df.loc[df["persona_cluster"]==cid, attr].mean()
             for cid in [0,1,2]]
    b = ax_p.bar([PERSONA_L[c] for c in [0,1,2]], means,
                 color=[PERSONA_C[c] for c in [0,1,2]],
                 edgecolor="white", linewidth=0.9, width=0.55, alpha=0.90)
    for bar,val in zip(b,means):
        ax_p.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                  f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax_p.set_title(f"Mean {label} by Persona",
                   color=attr_c, fontsize=11, fontweight="bold")
    ax_p.set_ylabel(label)
    for sp in ["left","bottom"]:
        ax_p.spines[sp].set_color(attr_c); ax_p.spines[sp].set_linewidth(1.3)

plt.savefig("outputs/17_user_personas.png", dpi=300, bbox_inches="tight")
plt.close(); print(" 17_user_personas.png")


# ══════════════════════════════════════════════════════════════
# SAVE MODELS + DATA
# ══════════════════════════════════════════════════════════════
joblib.dump(rf_reg,      "models/productivity_regressor.pkl")
joblib.dump(rf_clf,      "models/productivity_classifier.pkl")
joblib.dump(imputer,     "models/imputer.pkl")
joblib.dump(feature_cols,"models/feature_cols.pkl")

import json
metrics_out = {
    "ridge_r2":rf_r2,"rf_r2":rf_r2,"rf_mae":rf_mae,"rf_rmse":rf_rmse,
    "clf_accuracy":clf_acc,"cv_r2_mean":cv_scores.mean(),
    "cv_r2_std":cv_scores.std(),"cv_acc_mean":cv_acc.mean(),
    "cv_acc_std":cv_acc.std(),"silhouette":sil,
    "n_features":X.shape[1],"n_rows":len(df),
}
with open("models/metrics.json","w") as f: json.dump(metrics_out,f,indent=2)

df.to_csv("data/final_with_personas.csv", index=False)

print("\n══════════════════════════════════════════")
print("  ALL MODELLING CHARTS + MODELS SAVED ")
print("══════════════════════════════════════════")
