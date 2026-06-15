# QSR Loyalty Churn Predictor

Predicts which loyalty customers are likely to lapse — and segments active customers for targeted retention offers. Built as a proof-of-concept after noticing how much QSR churn analysis tends to focus on *who already left* rather than *who's about to*.

**[→ Live Dashboard](https://sirilahari10.github.io/qsr-churn-predictor)**

---

## Why I built this

Most churn models I've seen treat it as a binary outcome and stop there. What's actually useful for a team running a loyalty program is knowing *who to call* and *what to offer them*. So this project has two parts: a churn classifier to flag at-risk customers, and a segmentation layer that groups active customers so interventions can be personalized.

The QSR context felt relevant because loyalty programs in fast food generate really dense behavioral signals — app opens, offer redemptions, visit streaks — that aren't always used well.

## What it does

1. **Churn classification** (XGBoost) — predicts probability a loyalty customer will lapse in the next 30 days
2. **Customer segmentation** (K-Means, k=4) — groups active customers into actionable segments with suggested retention tactics
3. **Interactive dashboard** — single HTML file, no backend, deployable to GitHub Pages

## Results

| Metric | Score |
|---|---|
| ROC-AUC (test set) | 0.630 |
| 5-Fold CV AUC | 0.652 ± 0.012 |
| vs Logistic Regression baseline | +0.027 AUC |
| Accuracy | 65.8% |
| Avg Precision | 0.528 |

The AUC lift over logistic regression wasn't dramatic, but the XGBoost model caught non-linear interactions between visit drop and app engagement that the baseline missed — particularly for customers who stopped using the app 2-3 weeks before their last visit.

## Top churn signals

1. **Visit frequency trend** (biggest signal by far) — a drop >40% between 90-day windows
2. **Visits in the last 90 days** — absolute recency
3. **Customer satisfaction score** — survey non-response itself is a mild signal
4. **Loyalty tenure** — newer members churn at higher rates
5. **Offer redemption** — customers who never redeem are 2x more likely to churn

## Customer segments

| Segment | Size | Avg Visits/90d | Key trait | Suggested action |
|---|---|---|---|---|
| Champions | 1,653 | 7.6 | High tenure, high frequency | Double points, early access |
| Deal Seekers | 1,408 | 4.3 | High promo response (56%) | BOGO, value bundles |
| Casual Visitors | 3,011 | 4.0 | Low engagement | Streak rewards, visit milestones |

I tested k=3, 4, and 5. k=4 produced the most interpretable groupings, though Champions and Loyalists overlapped enough that I merged them in the final version.

## Limitations

- **Synthetic data** — built on simulated customers calibrated to QSR benchmarks, not real transaction data. Real POS and loyalty data would have seasonal patterns, menu-change effects, and geographic variance that aren't captured here.
- **No time series** — this is a snapshot model. A proper production version would use rolling windows and retrain on recent behavior.
- **Survey response bias** — 40% of customers have no satisfaction score. I filled with median, which is a rough proxy.
- **Segment stability** — K-Means segments can drift as customer mix changes. In production I'd monitor centroid drift quarterly.
- **Class imbalance** — the model has lower recall on churned customers (34%) than retained ones (84%). For a real retention campaign, I'd tune the decision threshold based on the cost of a false negative vs false positive.

## Stack

```
Python 3.11
├── pandas / numpy         — data wrangling & feature engineering
├── scikit-learn           — preprocessing, K-Means, logistic baseline
├── xgboost                — churn classification
└── Chart.js               — dashboard charts
```

## Run it

```bash
git clone https://github.com/sirilahari10/qsr-churn-predictor
cd qsr-churn-predictor
pip install -r requirements.txt

python generate_data.py   # generates data/customers.csv
python train_model.py     # trains model, outputs predictions + metrics
open index.html           # view dashboard locally
```

## About

**Siri Lahari Chava** — Data Scientist at Moderna, background in ML and computer vision.

[LinkedIn](https://linkedin.com/in/sirilahari) · [GitHub](https://github.com/sirilahari10)
