import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

print("SCRIPT STARTED")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"

def load_behavior():
    return pd.read_csv(DATA_DIR / "behavioral_summary.csv")

def train_clustering_model():
    behavior = load_behavior()

    features = [
        'saving_rate',
        'restaurant_ratio',
        'coffee_ratio',
        'shopping_ratio',
        'transport_ratio',
        'electronics_ratio',
        'entertainment_ratio',
        'weekend_spending_ratio',
        'planned_purchase_ratio',
        'online_purchase_ratio',
        'budget_adherence',
        'financial_score'
    ]

    X = behavior[features].dropna().copy()

    # ── Clean outliers ─────────────────────────────────────────────────
    print(f"\nBefore cleaning: {len(X)} rows")
    X = X[(X['saving_rate'] > -2) & (X['saving_rate'] <= 1)]
    X = X[(X['financial_score'] >= 0) & (X['financial_score'] <= 100)]
    ratio_cols = [c for c in features if 'ratio' in c]
    for col in ratio_cols:
        X = X[(X[col] >= 0) & (X[col] <= 1)]
    X = X[(X['budget_adherence'] >= 0) & (X['budget_adherence'] <= 1)]
    print(f"After cleaning : {len(X)} rows")
    print(f"Removed        : {len(behavior) - len(X)} outlier rows")

    # ── Scale ──────────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Find best K ────────────────────────────────────────────────────
    print("\n=== Finding Best Number of Clusters ===")
    scores = {}
    for k in range(2, 9):
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score  = silhouette_score(X_scaled, labels)
        scores[k] = score
        print(f"  K={k}  Silhouette Score: {score:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"\n✅ Best K: {best_k}")

    # ── Train final model ──────────────────────────────────────────────
    kmeans        = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    behavior_copy = behavior.loc[X.index].copy()
    behavior_copy['cluster'] = kmeans.fit_predict(X_scaled)

    # ── Cluster profiles ───────────────────────────────────────────────
    print("\n=== Cluster Profiles ===")
    profile = behavior_copy.groupby('cluster')[features].mean().round(3)
    print(profile.T.to_string())

    # ── Compare to existing personas ───────────────────────────────────
    print("\n=== Cluster vs Financial Personality ===")
    print(pd.crosstab(
        behavior_copy['financial_personality'],
        behavior_copy['cluster'],
        margins=True
    ))

    # ── Name clusters ──────────────────────────────────────────────────
    print("\n=== Cluster Naming ===")
    cluster_names = {}

    for cluster_id in range(best_k):
        row       = profile.loc[cluster_id]
        saving    = row['saving_rate']
        adherence = row['budget_adherence']
        score     = row['financial_score']

        cat_cols = ['restaurant_ratio','coffee_ratio','shopping_ratio',
                    'transport_ratio','electronics_ratio','entertainment_ratio']
        dominant = max(cat_cols, key=lambda c: row[c]).replace('_ratio','')

        if saving > 0.8 and adherence > 0.8:
            name = "Super Saver 💪"
        elif saving > 0.6 and adherence > 0.6 and score > 90:
            name = "High Scorer 🏆"
        elif saving > 0.6 and adherence > 0.6 and dominant == 'restaurant':
            name = "Foodie Saver 🍔"
        elif saving > 0.6 and adherence > 0.6 and dominant == 'entertainment':
            name = "Social Saver 🎉"
        elif saving > 0.6 and adherence > 0.6:
            name = "Balanced Saver 🟢"
        elif saving < -0.4 and adherence < 0.1 and dominant == 'electronics':
            name = "Tech Overspender 💸💻"
        elif saving < -0.4 and adherence < 0.1:
            name = "At-Risk Overspender 🔴"
        elif dominant == 'coffee' and row['coffee_ratio'] > 0.3:
            name = "Coffee Addict ☕"
        elif dominant == 'shopping' and row['shopping_ratio'] > 0.3:
            name = "Shopaholic 🛍️"
        elif dominant == 'electronics' and row['electronics_ratio'] > 0.5:
            name = "Tech Enthusiast 💻"
        elif dominant == 'entertainment' and row['entertainment_ratio'] > 0.2:
            name = "Social Spender 🎉"
        else:
            name = f"Moderate Spender ({dominant} leaning) 🟡"

        cluster_names[cluster_id] = name
        print(f"  Cluster {cluster_id} → '{name}'")
        print(f"    saving_rate={saving:.3f} | "
              f"budget_adherence={adherence:.3f} | "
              f"financial_score={score:.1f} | "
              f"dominant={dominant}")

    # ── Save everything ────────────────────────────────────────────────
    behavior_copy.to_csv(DATA_DIR / "behavioral_summary_clustered.csv", index=False)
    joblib.dump(kmeans,        MODELS_DIR / "clustering_model.pkl")
    joblib.dump(scaler,        MODELS_DIR / "clustering_scaler.pkl")
    joblib.dump(features,      MODELS_DIR / "clustering_features.pkl")
    joblib.dump(cluster_names, MODELS_DIR / "clustering_names.pkl")

    print("\n✅ All files saved.")
    return kmeans, scaler, behavior_copy, cluster_names


def predict_cluster(user_behavior_row: dict) -> dict:
    kmeans        = joblib.load(MODELS_DIR / "clustering_model.pkl")
    scaler        = joblib.load(MODELS_DIR / "clustering_scaler.pkl")
    features      = joblib.load(MODELS_DIR / "clustering_features.pkl")
    cluster_names = joblib.load(MODELS_DIR / "clustering_names.pkl")

    row        = pd.DataFrame([user_behavior_row])[features]
    row_scaled = scaler.transform(row)
    cluster_id = int(kmeans.predict(row_scaled)[0])

    return {
        "cluster_id"  : cluster_id,
        "cluster_name": cluster_names[cluster_id]
    }


if __name__ == "__main__":
    train_clustering_model()