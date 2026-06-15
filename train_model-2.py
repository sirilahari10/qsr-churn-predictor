"""
train_model.py

Two-part pipeline:
1. XGBoost churn classifier — predict which loyalty customers will lapse
2. K-Means segmentation — group active customers for targeted intervention

I started with logistic regression as a baseline, then moved to XGBoost after
seeing the interaction effects between visit frequency drop and app engagement
weren't being captured linearly. The lift was ~6 points AUC.

Siri Lahari Chava
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    accuracy_score, confusion_matrix, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

# ── Load & prep ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/customers.csv")

# Fill missing survey scores with median (40% didn't respond — common in QSR)
df["survey_score"] = df["survey_score"].fillna(df["survey_score"].median())

# Engineer a couple of features that felt meaningful from the data
df["visit_velocity"] = (df["visits_last_90d"] - df["visits_prior_90d"]) / (df["visits_prior_90d"] + 1)
df["engagement_score"] = df["app_opens_30d"] + df["offers_redeemed"] * 2

FEATURES = [
    "age", "tenure_months", "visits_last_90d", "visits_prior_90d",
    "avg_order_value", "app_opens_30d", "offers_redeemed", "loyalty_points",
    "points_redeemed_pct", "mobile_order_pct", "combo_meal_pct",
    "complaints", "survey_score", "promo_response_rate",
    "visit_velocity", "engagement_score"
]

X = df[FEATURES]
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Baseline: Logistic Regression ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train_s, y_train)
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_s)[:, 1])
print(f"Baseline (Logistic Regression) AUC: {lr_auc:.3f}")

# ── XGBoost ───────────────────────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.75,
    min_child_weight=3,
    gamma=0.1,
    eval_metric="logloss",
    random_state=42,
    verbosity=0
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print(f"\nXGBoost Results:")
print(f"  Accuracy : {acc:.1%}")
print(f"  ROC-AUC  : {auc:.3f}  (+{auc - lr_auc:.3f} vs baseline)")
print(f"  Avg Precision: {ap:.3f}")
print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(f"  5-Fold CV AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

# ── Feature importances ────────────────────────────────────────────────────────
feat_labels = {
    "visit_velocity": "Visit Frequency Trend",
    "visits_last_90d": "Visits (Last 90 Days)",
    "engagement_score": "App Engagement Score",
    "app_opens_30d": "App Opens (30 Days)",
    "survey_score": "Customer Satisfaction Score",
    "offers_redeemed": "Offers Redeemed",
    "promo_response_rate": "Promo Response Rate",
    "tenure_months": "Loyalty Tenure (Months)",
    "avg_order_value": "Avg Order Value",
    "loyalty_points": "Loyalty Points Balance",
    "complaints": "Complaint Count",
    "points_redeemed_pct": "Points Redemption Rate",
    "visits_prior_90d": "Visits (Prior 90 Days)",
    "mobile_order_pct": "Mobile Order %",
    "combo_meal_pct": "Combo Meal %",
    "age": "Customer Age"
}

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
importance_df["label"] = importance_df["feature"].map(feat_labels)

print("\nTop Churn Drivers:")
print(importance_df[["label", "importance"]].head(8).to_string(index=False))

# ── Predict on full dataset ────────────────────────────────────────────────────
df["churn_probability"] = model.predict_proba(X)[:, 1]
df["churn_prediction"] = model.predict(X)
df["risk_tier"] = pd.cut(
    df["churn_probability"],
    bins=[0, 0.35, 0.60, 1.0],
    labels=["Low", "Medium", "High"]
)

# ── K-Means Segmentation (active customers only) ───────────────────────────────
# Segment on customers NOT predicted to churn — who to invest in
active = df[df["churn_probability"] < 0.5].copy()

seg_features = ["visits_last_90d", "avg_order_value", "engagement_score",
                "tenure_months", "promo_response_rate"]
seg_scaled = StandardScaler().fit_transform(active[seg_features])

# Tried k=3,4,5 — k=4 gave the most interpretable segments
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
active["segment"] = kmeans.fit_predict(seg_scaled)

# Name segments based on their profiles
seg_profiles = active.groupby("segment").agg(
    size=("customer_id", "count"),
    avg_visits=("visits_last_90d", "mean"),
    avg_aov=("avg_order_value", "mean"),
    avg_engagement=("engagement_score", "mean"),
    avg_tenure=("tenure_months", "mean"),
    avg_promo=("promo_response_rate", "mean")
).round(2)

# Assign human-readable names
seg_names = {}
for seg_id, row in seg_profiles.iterrows():
    if row["avg_visits"] > 7 and row["avg_aov"] > 11:
        seg_names[seg_id] = "Champions"
    elif row["avg_visits"] > 5 and row["avg_engagement"] > 5:
        seg_names[seg_id] = "Loyalists"
    elif row["avg_promo"] > 0.3:
        seg_names[seg_id] = "Deal Seekers"
    else:
        seg_names[seg_id] = "Casual Visitors"

active["segment_name"] = active["segment"].map(seg_names)
print("\nCustomer Segments (active customers):")
print(seg_profiles)
print("\nSegment names:", seg_names)

# ── Save outputs ───────────────────────────────────────────────────────────────
df.to_csv("data/predictions.csv", index=False)
active.to_csv("data/segments.csv", index=False)

seg_summary = active.groupby("segment_name").agg(
    count=("customer_id", "count"),
    avg_visits=("visits_last_90d", "mean"),
    avg_aov=("avg_order_value", "mean"),
    avg_tenure=("tenure_months", "mean"),
    avg_promo=("promo_response_rate", "mean")
).round(2).reset_index()

metrics = {
    "accuracy": round(acc, 4),
    "roc_auc": round(auc, 4),
    "baseline_auc": round(lr_auc, 4),
    "avg_precision": round(ap, 4),
    "cv_auc_mean": round(cv_auc.mean(), 4),
    "cv_auc_std": round(cv_auc.std(), 4),
    "total_customers": len(df),
    "churn_rate": round(df["churned"].mean(), 4),
    "high_risk_count": int((df["risk_tier"] == "High").sum()),
    "medium_risk_count": int((df["risk_tier"] == "Medium").sum()),
    "low_risk_count": int((df["risk_tier"] == "Low").sum()),
    "confusion_matrix": cm.tolist(),
    "feature_importances": [
        {"feature": row["label"], "importance": round(float(row["importance"]), 4)}
        for _, row in importance_df.iterrows()
    ],
    "segments": seg_summary.to_dict(orient="records")
}

with open("data/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n✓ Saved predictions.csv, segments.csv, metrics.json")
print(f"  High risk customers: {metrics['high_risk_count']}")
print(f"  Active segments: {seg_summary['segment_name'].tolist()}")
