"""
Term-Deposit Predictor - Streamlit app for ML Assignment 2.

Loads the five pre-trained pipelines from model/, scores an uploaded test CSV
and shows evaluation metrics, a confusion matrix, a classification report and
a side-by-side comparison. No model is retrained here.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"

st.set_page_config(page_title="Term-Deposit Predictor", page_icon="🏦",
                   layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 3.5rem;}
div[data-testid="stMetric"] {
    background: #f4f7fb; border: 1px solid #dde5f0;
    border-radius: 10px; padding: 10px 14px;
}
.hero {font-size: 2.0rem; font-weight: 700; color: #1d3b6a; margin-bottom: 0;}
.sub  {color: #5a6b85; margin-top: 0.2rem;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_models():
    info = json.loads((MODEL_DIR / "feature_info.json").read_text())
    models = {name: joblib.load(MODEL_DIR / fname)
              for name, fname in info["models"].items()}
    return info, models


@st.cache_data
def load_training_metrics():
    return json.loads((MODEL_DIR / "metrics.json").read_text())


def positive_scores(pipe, X):
    try:
        return pipe.predict_proba(X)[:, 1]
    except AttributeError:
        return pipe.decision_function(X)


@st.cache_data(show_spinner="Scoring test data…")
def score_all(df_in: pd.DataFrame):
    """Predictions + positive-class scores for every model (cached per file)."""
    Xc = df_in[FEATURES]
    return {n: (p.predict(Xc), positive_scores(p, Xc)) for n, p in models.items()}


def compute_metrics(y_true, y_pred, scores):
    auc = (roc_auc_score(y_true, scores)
           if len(np.unique(y_true)) == 2 else float("nan"))
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def normalise_target(series):
    """Accept 0/1 or yes/no targets."""
    if series.dtype == object:
        series = series.str.strip().str.lower().map({"yes": 1, "no": 0})
    return series.astype(int)


info, models = load_models()
TARGET = info["target"]
FEATURES = info["numeric"] + info["categorical"]

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload test CSV", type="csv",
                                help="Must contain the 19 feature columns "
                                     "and the target column 'y'.")
    use_sample = st.checkbox("Use bundled test_data.csv instead",
                             value=uploaded is None)
    st.download_button("Download sample test_data.csv",
                       SAMPLE_CSV.read_bytes(), "test_data.csv", "text/csv")

    st.header("2 · Model")
    model_name = st.selectbox("Choose a classifier", list(models.keys()))

    st.header("3 · Threshold")
    threshold_on = st.toggle("Custom decision threshold", value=False,
                             help="Only for models with predict_proba "
                                  "(all except SVM).")
    threshold = st.slider("P(subscribe) ≥", 0.05, 0.95, 0.50, 0.05,
                          disabled=not threshold_on)

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown('<p class="hero">🏦 Term-Deposit Subscription Predictor</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub">UCI Bank Marketing · five pre-trained '
            'classifiers · evaluated on the test data you provide</p>',
            unsafe_allow_html=True)

if uploaded is not None and not use_sample:
    df = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample:
    df = pd.read_csv(SAMPLE_CSV)
    source = "bundled test_data.csv"
else:
    st.info("Upload a test CSV from the sidebar, or tick "
            "**Use bundled test_data.csv**.")
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"The CSV is missing required columns: {', '.join(missing)}")
    st.stop()

X = df[FEATURES]
y = normalise_target(df[TARGET])

with st.expander(f"Preview of {source} — {len(df):,} rows", expanded=False):
    st.dataframe(df.head(20), width="stretch")
    c1, c2 = st.columns(2)
    c1.metric("Subscribed (y=1)", f"{int(y.sum()):,}")
    c2.metric("Not subscribed (y=0)", f"{int((y == 0).sum()):,}")

# --------------------------------------------------------------------------- #
# Selected model
# --------------------------------------------------------------------------- #
pipe = models[model_name]
all_preds = score_all(df)
scores = all_preds[model_name][1]
has_proba = hasattr(pipe[-1], "predict_proba") and model_name != \
    "Support Vector Machine"
if threshold_on and has_proba:
    y_pred = (scores >= threshold).astype(int)
else:
    y_pred = all_preds[model_name][0]

tab_model, tab_compare, tab_about = st.tabs(
    ["📊 Selected model", "🏁 Compare all models", "ℹ️ About"])

with tab_model:
    st.subheader(f"{model_name}")
    if threshold_on and not has_proba:
        st.caption("SVM was trained without probability estimates, so the "
                   "default decision boundary is used.")
    m = compute_metrics(y, y_pred, scores)
    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"],
                    ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig, width="content")
        plt.close(fig)
    with right:
        st.markdown("**ROC curve**")
        fpr, tpr, _ = roc_curve(y, scores)
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.plot(fpr, tpr, color="#1d3b6a", lw=2,
                label=f"AUC = {m['AUC']:.3f}")
        ax.plot([0, 1], [0, 1], ls="--", color="#999")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right")
        st.pyplot(fig, width="content")
        plt.close(fig)

    st.markdown("**Classification report**")
    rep = classification_report(y, y_pred, target_names=["No (0)", "Yes (1)"],
                                output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).T
    rep_df.loc["accuracy", ["precision", "recall"]] = np.nan
    rep_df.loc["accuracy", "support"] = len(y)
    rep_df["support"] = rep_df["support"].round().astype(int)
    st.dataframe(rep_df.style.format("{:.3f}", subset=["precision", "recall",
                                                         "f1-score"],
                                     na_rep=""),
                 width="stretch")

    out = df.copy()
    out["predicted_y"] = y_pred
    st.download_button("Download predictions as CSV",
                       out.to_csv(index=False).encode(),
                       f"predictions_{model_name.replace(' ', '_')}.csv",
                       "text/csv")

with tab_compare:
    st.subheader("All five models on this test data")
    rows = {}
    for name, (pred, sc) in all_preds.items():
        rows[name] = compute_metrics(y, pred, sc)
    comp = pd.DataFrame(rows).T
    st.dataframe(comp.style.format("{:.4f}")
                 .highlight_max(axis=0, color="#cfe3ff"),
                 width="stretch")

    metric = st.radio("Plot metric", list(comp.columns), horizontal=True,
                      index=5)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    order = comp[metric].sort_values()
    ax.barh(order.index, order.values, color="#4a78b5")
    for i, v in enumerate(order.values):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    ax.set_xlabel(metric)
    ax.set_xlim(0, max(1.0, order.max() * 1.1))
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Highlighted cells mark the best model for each metric. "
               "Default thresholds are used for this comparison.")

with tab_about:
    st.markdown(f"""
**Dataset:** UCI Bank Marketing (bank-additional-full), 41,188 phone-call
records from a Portuguese bank, 2008–2010.
**Task:** predict whether a client subscribes to a term deposit (`y`).
**Features used ({len(FEATURES)}):** {", ".join(FEATURES)}.
`duration` is excluded because it is only known after the call ends.

All models are scikit-learn pipelines (StandardScaler + OneHotEncoder +
classifier) trained once in `train_models.py` on an 80 % stratified split
and loaded here from `model/`. Uploaded data is only scored, never trained on.
""")
    st.markdown("**Metrics recorded at training time (held-out 20 % split):**")
    st.dataframe(pd.DataFrame(load_training_metrics()).T,
                 width="stretch")
