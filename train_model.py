import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
"""
Train an XGBoost next-day ignition model with spatial cross-validation.
Produces metrics (AUC-PR, Brier), calibration and SHAP plots, and saves the trained model + metrics
"""

feat = pd.read_csv("data/cell_day_model_features.csv", parse_dates=["date"])
grid = pd.read_csv("data/grid_2km.csv")
feat = feat.merge(grid[["cell_id","cx","cy"]], on="cell_id", how="left")
X_cols = [c for c in feat.columns if c not in ["cell_id","date","y","cx","cy"]]
X = feat[X_cols].values
y = feat["y"].values
coords = feat[["cx","cy"]].values
km = KMeans(n_clusters=5, n_init=10, random_state=42).fit(coords)
groups = km.labels_
gkf = GroupKFold(n_splits=5)
probs = np.zeros(len(y))
for tr, va in gkf.split(X, y, groups=groups):
    ratio = y[tr].mean()
    spw = (1 - ratio) / max(ratio, 1e-6)
    model = XGBClassifier(max_depth=6, n_estimators=500, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, n_jobs=4, eval_metric="logloss", scale_pos_weight=spw, random_state=42)
    model.fit(X[tr], y[tr])
    probs[va] = model.predict_proba(X[va])[:,1]
ap = average_precision_score(y, probs)
br = brier_score_loss(y, probs)
m_true, m_pred = calibration_curve(y, probs, n_bins=10)
Path("figures").mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(5,4)); plt.plot(m_pred, m_true, marker="o"); plt.plot([0,1],[0,1], "--"); plt.title("Calibration"); plt.xlabel("Mean predicted"); plt.ylabel("Observed"); plt.tight_layout(); plt.savefig("figures/calibration.png", dpi=160); plt.close()
plt.figure(figsize=(5,4)); plt.hist(probs, bins=30); plt.title("Predicted probability histogram"); plt.xlabel("Risk"); plt.ylabel("Count"); plt.tight_layout(); plt.savefig("figures/pred_prob_hist.png", dpi=160); plt.close()
ratio = y.mean()
spw = (1 - ratio) / max(ratio, 1e-6)
final_model = XGBClassifier(max_depth=6, n_estimators=700, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, n_jobs=4, eval_metric="logloss", scale_pos_weight=spw, random_state=42)
final_model.fit(X, y)
explainer = shap.TreeExplainer(final_model)
idx = np.random.RandomState(42).choice(len(y), size=min(5000, len(y)), replace=False)
sv = explainer.shap_values(X[idx])
shap.summary_plot(sv, pd.DataFrame(X[idx], columns=X_cols), show=False)
plt.tight_layout(); plt.savefig("figures/shap_summary.png", dpi=160); plt.close()
imp = pd.Series(final_model.feature_importances_, index=X_cols).sort_values(ascending=False).head(20)
plt.figure(figsize=(6,5)); imp[::-1].plot(kind="barh"); plt.title("Top 20 feature importances"); plt.tight_layout(); plt.savefig("figures/feature_importances.png", dpi=160); plt.close()
Path("models").mkdir(parents=True, exist_ok=True)
joblib.dump(final_model, "models/xgb_nextday.pkl")
with open("models/metrics.txt","w") as f:
    f.write(f"AUC-PR: {ap:.3f}\nBrier: {br:.3f}\n")