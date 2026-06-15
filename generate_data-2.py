"""
generate_data.py
Generates synthetic but realistic QSR loyalty customer dataset.
Calibrated to fast food industry benchmarks (avg ticket $9-12, visit freq 2-3x/month).
"""

import pandas as pd
import numpy as np

np.random.seed(7)
N = 8000

def generate_qsr_customers(n=N):
    data = []

    for i in range(n):
        # Demographics
        age = int(np.clip(np.random.normal(34, 12), 18, 72))
        tenure_months = int(np.random.exponential(18))  # most customers are newer
        tenure_months = max(1, min(tenure_months, 60))

        # Behavioral signals
        visits_last_90d = np.random.poisson(lam=6 if tenure_months > 12 else 3)
        visits_prior_90d = np.random.poisson(lam=7 if tenure_months > 12 else 4)
        avg_order_value = round(np.random.normal(11.5, 3.2), 2)
        avg_order_value = max(4.0, avg_order_value)

        # App & loyalty engagement
        app_opens_30d = np.random.poisson(lam=4)
        offers_redeemed = np.random.poisson(lam=2)
        loyalty_points = int(np.random.exponential(800))
        points_redeemed_pct = round(np.random.uniform(0, 1), 2)

        # Order mix
        mobile_order_pct = round(np.random.beta(2, 3), 2)
        drive_thru_pct = round(np.random.beta(3, 2), 2)
        weekend_visit_pct = round(np.random.uniform(0.2, 0.6), 2)
        combo_meal_pct = round(np.random.beta(2, 2), 2)

        # Satisfaction signals
        complaints = np.random.poisson(lam=0.3)
        survey_score = int(np.clip(np.random.normal(7.8, 1.8), 1, 10)) if np.random.random() > 0.4 else None
        promo_response_rate = round(np.random.beta(2, 4), 2)

        # Churn logic — grounded in RFM and QSR industry patterns
        churn_prob = 0.22  # base ~22% annual churn
        visit_drop = (visits_prior_90d - visits_last_90d) / max(visits_prior_90d, 1)
        if visit_drop > 0.4: churn_prob += 0.25
        if visits_last_90d == 0: churn_prob += 0.30
        if app_opens_30d == 0: churn_prob += 0.12
        if offers_redeemed == 0: churn_prob += 0.08
        if complaints > 1: churn_prob += 0.10
        if survey_score and survey_score < 6: churn_prob += 0.12
        if tenure_months < 3: churn_prob += 0.10
        if avg_order_value < 7: churn_prob += 0.06
        if promo_response_rate < 0.1: churn_prob += 0.08
        churn_prob = min(churn_prob, 0.95)

        churned = int(np.random.random() < churn_prob)

        data.append({
            "customer_id": f"CUST-{i+1:05d}",
            "age": age,
            "tenure_months": tenure_months,
            "visits_last_90d": visits_last_90d,
            "visits_prior_90d": visits_prior_90d,
            "avg_order_value": avg_order_value,
            "app_opens_30d": app_opens_30d,
            "offers_redeemed": offers_redeemed,
            "loyalty_points": loyalty_points,
            "points_redeemed_pct": points_redeemed_pct,
            "mobile_order_pct": mobile_order_pct,
            "drive_thru_pct": drive_thru_pct,
            "weekend_visit_pct": weekend_visit_pct,
            "combo_meal_pct": combo_meal_pct,
            "complaints": complaints,
            "survey_score": survey_score,
            "promo_response_rate": promo_response_rate,
            "churned": churned
        })

    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_qsr_customers()
    df.to_csv("data/customers.csv", index=False)
    print(f"Dataset: {len(df)} customers")
    print(f"Churn rate: {df['churned'].mean():.1%}")
    print(f"Avg order value: ${df['avg_order_value'].mean():.2f}")
    print(f"Avg visits/90d: {df['visits_last_90d'].mean():.1f}")
