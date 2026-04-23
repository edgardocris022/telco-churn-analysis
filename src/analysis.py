from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ──────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────

def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Run the script from the project root or place the CSV in the data/ folder."
        )
    return pd.read_csv(csv_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-0.1, 12, 24, 48, 72, np.inf],
        labels=["0-1y", "1-2y", "2-4y", "4-6y", "6y+"],
        include_lowest=True,
    )
    return df


# ──────────────────────────────────────────────
# EDA
# ──────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", palette="husl")
    plt.rcParams["figure.figsize"] = (12, 6)

    ax = sns.countplot(data=df, x="Churn")
    ax.set_title("Churn Distribution")
    plt.tight_layout()
    plt.show()

    ax = sns.countplot(data=df, x="Contract", hue="Churn")
    ax.set_title("Churn by Contract Type")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

    ax = sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
    ax.set_title("Monthly Charges vs Churn")
    plt.tight_layout()
    plt.show()

    num_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df.columns]
    if num_cols:
        corr = df[num_cols].corr(numeric_only=True)
        ax = sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title("Correlation Matrix (numeric subset)")
        plt.tight_layout()
        plt.show()

    ax = sns.histplot(data=df, x="tenure", bins=30)
    ax.set_title("Customer Tenure Distribution")
    plt.tight_layout()
    plt.show()

    if "tenure_group" in df.columns:
        plt.figure(figsize=(10, 5))
        ax = sns.countplot(
            data=df,
            x="tenure_group",
            order=[g for g in ["0-1y", "1-2y", "2-4y", "4-6y", "6y+"]
                   if g in df["tenure_group"].astype(str).unique()],
            hue="Churn",
        )
        ax.set_title("Churn by Tenure Group")
        ax.set_xlabel("Customer Tenure")
        ax.set_ylabel("Number of Customers")
        plt.tight_layout()
        plt.show()


# ──────────────────────────────────────────────
# PIPELINE BUILDER (compartido por ambos modelos)
# ──────────────────────────────────────────────

def build_preprocessor(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor, cat_cols, num_cols


def prepare_data(df: pd.DataFrame):
    required = {"Churn", "customerID"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    data = df.drop(columns=["customerID"]).copy()

    churn_map = {"No": 0, "Yes": 1}
    data["Churn"] = data["Churn"].map(churn_map)
    if data["Churn"].isna().any():
        bad = sorted(set(df["Churn"].dropna().unique()) - set(churn_map.keys()))
        raise ValueError(f"Unexpected Churn values: {bad}")

    X = data.drop(columns=["Churn"])
    y = data["Churn"].astype(int)
    return X, y


# ──────────────────────────────────────────────
# MODELO 1 — LOGISTIC REGRESSION
# ──────────────────────────────────────────────

def train_logistic_regression(X_train, X_test, y_train, y_test, preprocessor) -> None:
    print("=" * 50)
    print("MODEL: Logistic Regression")
    print("=" * 50)

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=2000, n_jobs=None)),
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))


# ──────────────────────────────────────────────
# MODELO 2 — RANDOM FOREST
# ──────────────────────────────────────────────

def train_random_forest(X_train, X_test, y_train, y_test, preprocessor, cat_cols, num_cols) -> None:
    print("=" * 50)
    print("MODEL: Random Forest")
    print("=" * 50)

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # ── Feature Importance ──
    ohe = clf.named_steps["preprocess"].named_transformers_["cat"]
    ohe_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + ohe_feature_names

    importances = clf.named_steps["model"].feature_importances_
    feat_df = (
        pd.DataFrame({"feature": all_feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_df, x="importance", y="feature", palette="viridis")
    plt.title("Top 15 Feature Importances — Random Forest")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    print("\nTop 15 Features:")
    print(feat_df.to_string(index=False))


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    df = load_dataset(csv_path)
    print(df.head())
    print(df.info())
    print(df.isnull().sum())

    df = clean_data(df)
    df = add_features(df)

    run_eda(df)

    X, y = prepare_data(df)
    preprocessor, cat_cols, num_cols = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    train_logistic_regression(X_train, X_test, y_train, y_test, preprocessor)
    train_random_forest(X_train, X_test, y_train, y_test, preprocessor, cat_cols, num_cols)


if __name__ == "__main__":
    main()