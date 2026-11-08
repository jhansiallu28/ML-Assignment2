"""
ML Assignment 2 - Training script
Dataset : UCI Bank Marketing (bank-additional-full), 41,188 rows
Target  : y  (1 = client subscribed to a term deposit, 0 = did not)

Trains five classifiers on the same train split, evaluates them on a held-out
test split, saves every fitted pipeline to model/ and writes the held-out
split to test_data.csv so the Streamlit app can score it without retraining.

Run:  python train_models.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "bank_marketing_full.csv"
MODEL_DIR = ROOT / "model"
TARGET = "y"

# 'duration' (call length) is only known after the call ends, so it leaks the
# outcome. The UCI dataset notes recommend dropping it for a realistic model.
DROP_COLS = ["duration"]

CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "day_of_week", "poutcome"]
NUMERIC = ["age", "campaign", "pdays", "previous", "emp_var_rate",
           "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"]
FEATURES = NUMERIC + CATEGORICAL


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL),
    ])


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0,
                                                  random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=6,
                                                min_samples_leaf=50,
                                                random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=25, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", C=1.0, gamma="scale",
                                      random_state=SEED),
    }


def positive_scores(pipe, X):
    """Score for the positive class, used for AUC."""
    if hasattr(pipe, "predict_proba"):
        try:
            return pipe.predict_proba(X)[:, 1]
        except AttributeError:  # SVC without probability=True
            pass
    return pipe.decision_function(X)


def evaluate(pipe, X, y) -> dict:
    y_pred = pipe.predict(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, positive_scores(pipe, X)),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def file_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_") + ".joblib"


def main():
    df = pd.read_csv(DATA_PATH).drop(columns=DROP_COLS)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1] - 1} features")
    print("Class balance:", df[TARGET].value_counts().to_dict())

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    # Held-out split saved for the Streamlit app (features + target)
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    MODEL_DIR.mkdir(exist_ok=True)
    results = {}
    for name, clf in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_s = time.time() - t0
        joblib.dump(pipe, MODEL_DIR / file_name(name), compress=3)
        results[name] = {k: round(float(v), 4)
                         for k, v in evaluate(pipe, X_test, y_test).items()}
        print(f"{name:<24} fit {fit_s:6.1f}s  {results[name]}")

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(MODEL_DIR / "feature_info.json", "w") as f:
        json.dump({"target": TARGET, "numeric": NUMERIC,
                   "categorical": CATEGORICAL, "dropped": DROP_COLS,
                   "models": {n: file_name(n) for n in results}}, f, indent=2)

    table = pd.DataFrame(results).T
    print("\n", table.to_markdown(floatfmt=".4f"))


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
