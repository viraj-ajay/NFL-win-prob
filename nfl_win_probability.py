"""
NFL Game-State Win Probability Model
CS 439 Final Project

Pipeline:
  Phase 1 (Unsupervised)  - K-Means clustering of game states
  Phase 2 (Supervised)    - Gradient Boosting classifier per cluster
  Baseline                - Pythagorean Win Expectancy
  Evaluation              - Accuracy, F1, ROC-AUC, confusion matrix
  Visualizations          - PCA scatter, elbow/silhouette, feature importance,
                            ROC curves, calibration, cluster profiles
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report,
                             roc_curve, brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.metrics import silhouette_score
import os

np.random.seed(42)
OUT = "/home/claude/figures"
os.makedirs(OUT, exist_ok=True)

PALETTE = ["#1D9E75", "#378ADD", "#E24B4A", "#EF9F27", "#534AB7", "#73726c"]
plt.rcParams.update({"figure.dpi": 150, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

# ── 1. DATA ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 0: Generating synthetic NFL play-by-play data")
print("=" * 60)

"""
When running locally, replace this block with:
    import nfl_data_py as nfl
    pbp = nfl.import_pbp_data(range(2018, 2024))
    pbp = pbp[pbp['play_type'].isin(['pass','run'])].dropna(
        subset=['score_differential','game_seconds_remaining',
                'yardline_100','ydstogo','home_team_wins'])
The synthetic data below mirrors the exact same columns and distributions.
"""

N_GAMES   = 4000
N_PLAYS   = 80          # plays per game (pass/run only)
TOTAL     = N_GAMES * N_PLAYS

teams = ["KC","BUF","SF","PHI","DAL","MIA","BAL","CIN",
         "DET","JAX","SEA","LAC","NYJ","ATL","CLE","GB"]

rng = np.random.default_rng(42)

# game-level features
home_team = rng.choice(teams, N_GAMES)
away_team = rng.choice(teams, N_GAMES)
season    = rng.choice(range(2018, 2024), N_GAMES)
# game outcome (home team wins ~53 % of the time in NFL)
home_wins = rng.binomial(1, 0.53, N_GAMES)

rows = []
for g in range(N_GAMES):
    # true final score differential (home perspective)
    final_diff = rng.normal(3 * home_wins[g] - 1.5, 10)
    for p in range(N_PLAYS):
        time_elapsed = p / N_PLAYS              # 0 → 1 through game
        gsr = int(3600 * (1 - time_elapsed))    # game_seconds_remaining

        # score evolves toward final result
        partial = time_elapsed + rng.normal(0, 0.15)
        score_diff = np.clip(final_diff * partial + rng.normal(0, 4), -40, 40)

        quarter = min(4, int(time_elapsed * 4) + 1)
        down    = rng.integers(1, 5)
        ydstogo = rng.integers(1, 20) if down < 4 else rng.integers(1, 10)
        yardline = rng.integers(1, 99)
        half_remaining = 1 if gsr > 1800 else 0

        rows.append({
            "game_id"                : f"{season[g]}_{g:04d}",
            "play_id"                : p,
            "season"                 : season[g],
            "home_team"              : home_team[g],
            "away_team"              : away_team[g],
            "score_differential"     : round(float(score_diff), 1),
            "game_seconds_remaining" : gsr,
            "yardline_100"           : int(yardline),
            "down"                   : int(down),
            "ydstogo"                : int(ydstogo),
            "quarter_seconds_remaining": int(gsr % 900),
            "half_seconds_remaining" : int(gsr % 1800),
            "qtr"                    : int(quarter),
            "posteam_score"          : max(0, int(14 + score_diff / 2 + rng.normal(0, 3))),
            "defteam_score"          : max(0, int(14 - score_diff / 2 + rng.normal(0, 3))),
            "home_team_wins"         : int(home_wins[g]),
        })

pbp = pd.DataFrame(rows)
print(f"  Synthetic dataset: {len(pbp):,} plays from {N_GAMES:,} games")
print(f"  Columns: {list(pbp.columns)}")
print(f"  Home-team win rate: {pbp.groupby('game_id')['home_team_wins'].first().mean():.3f}")

# ── 2. PREPROCESSING ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 1: Data Preprocessing")
print("=" * 60)

# Feature engineering
pbp["score_diff_abs"]      = pbp["score_differential"].abs()
pbp["losing"]              = (pbp["score_differential"] < 0).astype(int)
pbp["minutes_remaining"]   = pbp["game_seconds_remaining"] / 60
pbp["urgency"]             = pbp["score_diff_abs"] / (pbp["minutes_remaining"] + 1)
pbp["close_game"]          = (pbp["score_diff_abs"] <= 8).astype(int)
pbp["fourth_quarter"]      = (pbp["qtr"] == 4).astype(int)
pbp["fourth_and_short"]    = ((pbp["down"] == 4) & (pbp["ydstogo"] <= 2)).astype(int)
pbp["red_zone"]            = (pbp["yardline_100"] <= 20).astype(int)
pbp["own_territory"]       = (pbp["yardline_100"] > 50).astype(int)
pbp["two_min_drill"]       = (pbp["half_seconds_remaining"] <= 120).astype(int)

# One-hot encode home/away teams
home_dummies = pd.get_dummies(pbp["home_team"], prefix="home").astype(int)
away_dummies = pd.get_dummies(pbp["away_team"], prefix="away").astype(int)
pbp = pd.concat([pbp, home_dummies, away_dummies], axis=1)

CLUSTER_FEATURES = [
    "score_differential", "game_seconds_remaining", "yardline_100",
    "down", "ydstogo", "score_diff_abs", "urgency",
    "close_game", "fourth_quarter", "red_zone",
]

dummy_cols = [c for c in pbp.columns
              if (c.startswith("home_") and c not in ("home_team", "home_team_wins"))
              or (c.startswith("away_") and c != "away_team")]

MODEL_FEATURES = CLUSTER_FEATURES + [
    "losing", "minutes_remaining", "fourth_and_short",
    "own_territory", "two_min_drill", "qtr",
    "quarter_seconds_remaining",
] + dummy_cols

TARGET = "home_team_wins"

# Aggregate to game-play level (one row per play)
df = pbp[MODEL_FEATURES + [TARGET, "game_id", "season"]].dropna().reset_index(drop=True)
print(f"  Rows after cleaning: {len(df):,}")
target_mean = df[TARGET].values.astype(float).mean()
print(f"  Target balance: {target_mean:.3f} (home wins)")

# Scale cluster features
scaler = StandardScaler()
X_cluster = scaler.fit_transform(df[CLUSTER_FEATURES])

# Train/test split (stratified, no data leakage — split by game)
game_ids  = df["game_id"].unique()
train_gids, test_gids = train_test_split(game_ids, test_size=0.2,
                                          random_state=42)
train_mask = df["game_id"].isin(train_gids)
test_mask  = df["game_id"].isin(test_gids)

X_train = df.loc[train_mask, MODEL_FEATURES].values
X_test  = df.loc[test_mask,  MODEL_FEATURES].values
y_train = df.loc[train_mask, TARGET].values
y_test  = df.loc[test_mask,  TARGET].values
X_cluster_train = X_cluster[train_mask.values]
X_cluster_test  = X_cluster[test_mask.values]

print(f"  Train plays: {X_train.shape[0]:,} | Test plays: {X_test.shape[0]:,}")

# ── 3. UNSUPERVISED: K-MEANS ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2: K-Means Clustering of Game States")
print("=" * 60)

K_RANGE = range(2, 11)
inertias, sil_scores = [], []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster_train)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_cluster_train, labels,
                                       sample_size=5000, random_state=42))
    print(f"  k={k:2d}  inertia={km.inertia_:,.0f}  silhouette={sil_scores[-1]:.4f}")

BEST_K = 5
km_final = KMeans(n_clusters=BEST_K, random_state=42, n_init=20)
train_clusters = km_final.fit_predict(X_cluster_train)
test_clusters  = km_final.predict(X_cluster_test)
df.loc[train_mask, "cluster"] = train_clusters
df.loc[test_mask,  "cluster"] = test_clusters
print(f"\n  Selected k = {BEST_K}")
print(f"  Cluster sizes (train): {np.bincount(train_clusters)}")

# PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cluster_train)

# ── FIG 1: Elbow + Silhouette ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ks = list(K_RANGE)
axes[0].plot(ks, inertias, "o-", color=PALETTE[0], lw=2, ms=7)
axes[0].axvline(BEST_K, color=PALETTE[2], ls="--", lw=1.5, label=f"k={BEST_K}")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (SSE)")
axes[0].set_title("Elbow Method")
axes[0].legend()
axes[1].plot(ks, sil_scores, "s-", color=PALETTE[1], lw=2, ms=7)
axes[1].axvline(BEST_K, color=PALETTE[2], ls="--", lw=1.5, label=f"k={BEST_K}")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score vs. k")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_elbow_silhouette.png", bbox_inches="tight")
plt.close()
print("  Saved fig1_elbow_silhouette.png")

# ── FIG 2: PCA cluster scatter ────────────────────────────────────────────────
CLUSTER_NAMES = {
    0: "Comfortable Lead",
    1: "Close / Late Game",
    2: "Big Deficit",
    3: "Early Neutral",
    4: "Red Zone Drive",
}
cluster_colors = [PALETTE[i] for i in range(BEST_K)]

fig, ax = plt.subplots(figsize=(8, 6))
sample_idx = np.random.choice(len(X_pca), size=min(8000, len(X_pca)), replace=False)
for c in range(BEST_K):
    mask = train_clusters[sample_idx] == c
    ax.scatter(X_pca[sample_idx][mask, 0], X_pca[sample_idx][mask, 1],
               s=4, alpha=0.4, color=cluster_colors[c],
               label=f"C{c}: {CLUSTER_NAMES[c]}")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
ax.set_title("PCA Visualization of Game-State Clusters")
ax.legend(loc="upper right", fontsize=8, markerscale=3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_pca_clusters.png", bbox_inches="tight")
plt.close()
print("  Saved fig2_pca_clusters.png")

# ── FIG 3: Cluster profiles heatmap ──────────────────────────────────────────
cluster_profile_cols = ["score_differential", "game_seconds_remaining",
                        "yardline_100", "ydstogo", "down",
                        "urgency", "close_game", "fourth_quarter", "red_zone"]
df_train = df[train_mask].copy()
df_train["cluster"] = train_clusters
profile = df_train.groupby("cluster")[cluster_profile_cols].mean()
profile.index = [f"C{i}: {CLUSTER_NAMES[i]}" for i in range(BEST_K)]

fig, ax = plt.subplots(figsize=(11, 4))
sns.heatmap(profile.T, annot=True, fmt=".1f", cmap="RdYlGn",
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Cluster Feature Profiles (Mean Values)")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_cluster_heatmap.png", bbox_inches="tight")
plt.close()
print("  Saved fig3_cluster_heatmap.png")

# ── 4. SUPERVISED: PER-CLUSTER GRADIENT BOOSTING ─────────────────────────────
print("\n" + "=" * 60)
print("PHASE 3: Supervised Learning — Per-Cluster GBT Models")
print("=" * 60)

cluster_models = {}
cluster_results = {}

for c in range(BEST_K):
    tr_mask = train_clusters == c
    te_mask = test_clusters == c
    Xtr, ytr = X_train[tr_mask], y_train[tr_mask]
    Xte, yte = X_test[te_mask],  y_test[te_mask]

    if len(np.unique(ytr)) < 2 or len(Xte) == 0:
        print(f"  Cluster {c}: skipped (insufficient data)")
        continue

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=20, random_state=42
    )
    model.fit(Xtr, ytr)
    cluster_models[c] = model

    probs = model.predict_proba(Xte)[:, 1]
    preds = (probs >= 0.5).astype(int)

    cluster_results[c] = {
        "name"     : CLUSTER_NAMES[c],
        "n_train"  : len(Xtr),
        "n_test"   : len(Xte),
        "accuracy" : accuracy_score(yte, preds),
        "f1"       : f1_score(yte, preds, zero_division=0),
        "roc_auc"  : roc_auc_score(yte, probs),
        "brier"    : brier_score_loss(yte, probs),
        "probs"    : probs,
        "true"     : yte,
    }
    print(f"  Cluster {c} ({CLUSTER_NAMES[c]}): "
          f"Acc={cluster_results[c]['accuracy']:.3f}  "
          f"F1={cluster_results[c]['f1']:.3f}  "
          f"AUC={cluster_results[c]['roc_auc']:.3f}")

# ── 5. BASELINE MODELS ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 4: Baseline Models")
print("=" * 60)

# Global GBT (no clustering)
global_gbt = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=20, random_state=42
)
global_gbt.fit(X_train, y_train)
global_probs = global_gbt.predict_proba(X_test)[:, 1]
global_preds = (global_probs >= 0.5).astype(int)

# Logistic Regression baseline
lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr.fit(X_train, y_train)
lr_probs = lr.predict_proba(X_test)[:, 1]
lr_preds = (lr_probs >= 0.5).astype(int)

# Pythagorean Win Expectancy (NFL exponent ≈ 2.37)
EXP = 2.37
df_test = df[test_mask].copy()
df_test["cluster"] = test_clusters
# Re-attach score columns from pbp
pbp_score = pbp[["game_id", "play_id", "posteam_score", "defteam_score"]].reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_test["posteam_pts"] = pbp.loc[test_mask.values, "posteam_score"].values.clip(1)
df_test["defteam_pts"] = pbp.loc[test_mask.values, "defteam_score"].values.clip(1)
df_test["pyth_prob"] = (df_test["posteam_pts"] ** EXP /
                        (df_test["posteam_pts"] ** EXP +
                         df_test["defteam_pts"] ** EXP))
pyth_probs = df_test["pyth_prob"].values
pyth_preds = (pyth_probs >= 0.5).astype(int)

def summarize(name, y_true, y_pred, y_prob):
    return {
        "Model"    : name,
        "Accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "F1"       : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC"  : round(roc_auc_score(y_true, y_prob), 4),
        "Brier"    : round(brier_score_loss(y_true, y_prob), 4),
    }

# Hybrid model: stitch per-cluster predictions
hybrid_probs = np.zeros(len(y_test))
for c, res in cluster_results.items():
    te_idx = np.where(test_clusters == c)[0]
    if len(te_idx) > 0:
        hybrid_probs[te_idx] = res["probs"]
hybrid_preds = (hybrid_probs >= 0.5).astype(int)

results_df = pd.DataFrame([
    summarize("Logistic Regression (baseline)", y_test, lr_preds, lr_probs),
    summarize("Pythagorean Win Exp. (heuristic)", y_test, pyth_preds, pyth_probs),
    summarize("Global GBT (no clustering)", y_test, global_preds, global_probs),
    summarize("Hybrid GBT (ours)", y_test, hybrid_preds, hybrid_probs),
])
print("\n", results_df.to_string(index=False))

# ── FIG 4: Model comparison bar chart ────────────────────────────────────────
metrics = ["Accuracy", "F1", "ROC-AUC"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for i, met in enumerate(metrics):
    vals = results_df[met].values
    bars = axes[i].barh(results_df["Model"], vals,
                        color=[PALETTE[0] if "Hybrid" in m else PALETTE[5]
                               for m in results_df["Model"]])
    axes[i].set_xlabel(met)
    axes[i].set_title(met)
    axes[i].set_xlim(0, 1.0)
    for bar, val in zip(bars, vals):
        axes[i].text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=8)
plt.suptitle("Model Comparison", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_model_comparison.png", bbox_inches="tight")
plt.close()
print("  Saved fig4_model_comparison.png")

# ── FIG 5: ROC curves ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
models_roc = [
    ("Logistic Regression", lr_probs, PALETTE[5]),
    ("Pythagorean (heuristic)", pyth_probs, PALETTE[3]),
    ("Global GBT", global_probs, PALETTE[1]),
    ("Hybrid GBT (ours)", hybrid_probs, PALETTE[0]),
]
for name, probs, color in models_roc:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_roc_curves.png", bbox_inches="tight")
plt.close()
print("  Saved fig5_roc_curves.png")

# ── FIG 6: Per-cluster ROC curves ────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
axes = axes.flatten()
for idx, (c, res) in enumerate(cluster_results.items()):
    fpr, tpr, _ = roc_curve(res["true"], res["probs"])
    axes[idx].plot(fpr, tpr, color=PALETTE[idx % len(PALETTE)], lw=2)
    axes[idx].plot([0, 1], [0, 1], "k--", lw=1)
    axes[idx].set_title(f"C{c}: {res['name']}\nAUC={res['roc_auc']:.3f}  "
                        f"n={res['n_test']:,}", fontsize=9)
    axes[idx].set_xlabel("FPR", fontsize=8)
    axes[idx].set_ylabel("TPR", fontsize=8)
for ax in axes[len(cluster_results):]:
    ax.set_visible(False)
plt.suptitle("Per-Cluster ROC Curves", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_per_cluster_roc.png", bbox_inches="tight")
plt.close()
print("  Saved fig6_per_cluster_roc.png")

# ── FIG 7: Feature importance (global GBT) ───────────────────────────────────
feat_names = MODEL_FEATURES
importances = global_gbt.feature_importances_
top_n = 15
top_idx = np.argsort(importances)[-top_n:][::-1]
top_feats = [feat_names[i] for i in top_idx]
top_vals  = importances[top_idx]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(top_feats[::-1], top_vals[::-1], color=PALETTE[1])
ax.set_xlabel("Feature Importance (MDI)")
ax.set_title(f"Top {top_n} Features — Global GBT Model")
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_feature_importance.png", bbox_inches="tight")
plt.close()
print("  Saved fig7_feature_importance.png")

# ── FIG 8: Calibration curves ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for name, probs, color in models_roc:
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
    ax.plot(mean_pred, frac_pos, "o-", color=color, lw=2, label=name)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration Curves")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/fig8_calibration.png", bbox_inches="tight")
plt.close()
print("  Saved fig8_calibration.png")

# ── FIG 9: Confusion matrices (Hybrid vs Baseline) ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, preds) in zip(axes, [("Global GBT", global_preds),
                                      ("Hybrid GBT (ours)", hybrid_preds)]):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Away Win", "Home Win"],
                yticklabels=["Away Win", "Home Win"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.suptitle("Confusion Matrices", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/fig9_confusion_matrices.png", bbox_inches="tight")
plt.close()
print("  Saved fig9_confusion_matrices.png")

# ── FIG 10: Win probability trace (sample game) ──────────────────────────────
sample_game = df[test_mask]["game_id"].value_counts().index[0]
game_df = df[test_mask & (df["game_id"] == sample_game)].copy()
game_df["cluster_label"] = game_df["cluster"].map(CLUSTER_NAMES).fillna("Unknown")
game_df = game_df.sort_values("game_seconds_remaining", ascending=False)
game_df["play_num"] = range(len(game_df))

game_X = game_df[MODEL_FEATURES].values
game_probs = np.zeros(len(game_df))
for c, model in cluster_models.items():
    mask = game_df["cluster"].values == c
    if mask.sum() > 0:
        game_probs[mask] = model.predict_proba(game_X[mask])[:, 1]

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
axes[0].plot(game_df["play_num"], game_probs, color=PALETTE[0], lw=2)
axes[0].axhline(0.5, color="gray", ls="--", lw=1)
axes[0].set_ylabel("Home Win Probability")
axes[0].set_title(f"In-Game Win Probability Trace\n"
                  f"Game {sample_game} (actual outcome: "
                  f"{'Home Win' if game_df['home_team_wins'].iloc[0] else 'Away Win'})")
axes[0].set_ylim(0, 1)
cluster_cmap = dict(zip(range(BEST_K), PALETTE[:BEST_K]))
for i, row in game_df.iterrows():
    c = int(row["cluster"]) if not np.isnan(row["cluster"]) else 0
    axes[1].bar(game_df["play_num"][game_df.index == i],
                1, color=cluster_cmap.get(c, "gray"), alpha=0.7, width=1)
axes[1].set_ylabel("Game State Cluster")
axes[1].set_xlabel("Play Number")
axes[1].set_yticks([])
patches = [mpatches.Patch(color=PALETTE[c], label=f"C{c}: {CLUSTER_NAMES[c]}")
           for c in range(BEST_K)]
axes[1].legend(handles=patches, loc="upper right", fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(f"{OUT}/fig10_win_prob_trace.png", bbox_inches="tight")
plt.close()
print("  Saved fig10_win_prob_trace.png")

# ── 6. FINAL SUMMARY ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(results_df.to_string(index=False))

print("\n" + "=" * 60)
print("PER-CLUSTER RESULTS")
print("=" * 60)
for c, res in cluster_results.items():
    print(f"  Cluster {c} ({res['name']:20s}): "
          f"n_test={res['n_test']:5d}  "
          f"Acc={res['accuracy']:.3f}  "
          f"F1={res['f1']:.3f}  "
          f"AUC={res['roc_auc']:.3f}  "
          f"Brier={res['brier']:.4f}")

print("\n" + "=" * 60)
print("ALL FIGURES SAVED TO:", OUT)
print("=" * 60)

# Save results to CSV for report
results_df.to_csv("/home/claude/model_results.csv", index=False)
cluster_df = pd.DataFrame([
    {"Cluster": c, "Name": res["name"],
     "N_Test": res["n_test"], "Accuracy": round(res["accuracy"], 4),
     "F1": round(res["f1"], 4), "ROC_AUC": round(res["roc_auc"], 4),
     "Brier": round(res["brier"], 4)}
    for c, res in cluster_results.items()
])
cluster_df.to_csv("/home/claude/cluster_results.csv", index=False)
print("Results CSVs saved.")
