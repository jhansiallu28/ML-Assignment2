# Term-Deposit Subscription Prediction — ML Assignment 2

**Live app:** `<paste your Streamlit Community Cloud URL here>`
**GitHub repo:** `<paste your GitHub repository URL here>`

---

## a. Problem statement

A Portuguese retail bank ran phone-based marketing campaigns to sell term
deposits. Each call costs agent time, and only about 1 in 9 contacted clients
subscribe. The goal is to build a **binary classifier** that predicts, *before
the call is made*, whether a client will subscribe (`y = 1`) or not (`y = 0`).
The bank can then rank clients and call the most promising ones first.

Five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbours,
Gaussian Naive Bayes and an RBF Support Vector Machine) are trained on the
same data and compared on Accuracy, AUC, Precision, Recall, F1 and the
Matthews Correlation Coefficient (MCC). The models are served through an
interactive Streamlit app.

## b. Dataset description

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository: *Bank Marketing* (`bank-additional-full.csv`), Moro, Cortez & Rita (2014). https://archive.ics.uci.edu/dataset/222/bank+marketing |
| Instances | 41,188 |
| Features in the raw file | 20 (+ target) |
| Features used | **19** (`duration` dropped, see below) |
| Target | `y`: 1 = subscribed to a term deposit, 0 = did not |
| Class balance | 36,548 "no" (88.7 %) / 4,640 "yes" (11.3 %): imbalanced |
| Missing values | None as NaN; some categoricals use the level `unknown`, kept as its own category |

**Features**

| Group | Columns |
|---|---|
| Client (categorical) | `job`, `marital`, `education`, `default`, `housing`, `loan` |
| Last contact (categorical) | `contact`, `month`, `day_of_week` |
| Campaign (numeric / categorical) | `campaign` (contacts this campaign), `pdays` (days since previous campaign, 999 = never), `previous` (contacts before), `poutcome` (previous outcome) |
| Client (numeric) | `age` |
| Social and economic context (numeric) | `emp_var_rate`, `cons_price_idx`, `cons_conf_idx`, `euribor3m`, `nr_employed` |

**Why `duration` is dropped:** call duration is only known after the call
ends, and a duration of 0 always means `y = 0`. The UCI documentation says it
should be discarded for a realistic predictive model, so it is excluded here.

**Preprocessing:** `StandardScaler` on the 9 numeric columns and
`OneHotEncoder(handle_unknown="ignore")` on the 10 categorical columns,
wrapped with each classifier in a single scikit-learn `Pipeline`. Data was
split 80 / 20 with stratification on `y` (`random_state=42`):
32,950 training rows and **8,238 test rows** (saved as `test_data.csv`).

## c. GitHub repository

`<paste your GitHub repository URL here>`

```
bank-marketing-classifier/
├── app.py                  # Streamlit app (loads saved models, no retraining)
├── train_models.py         # trains + evaluates all 5 models, saves to model/
├── requirements.txt
├── README.md
├── test_data.csv           # held-out 20 % test split (features + target y)
├── data/
│   └── bank_marketing_full.csv   # full UCI dataset used for training
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── support_vector_machine.joblib
│   ├── metrics.json        # test-set metrics written at training time
│   └── feature_info.json   # feature lists + model file map used by app.py
└── .streamlit/config.toml  # app theme
```

**Run locally**

```bash
pip install -r requirements.txt
python train_models.py      # optional: re-creates model/ and test_data.csv
streamlit run app.py
```

## d. Models used

| Model | Key settings |
|---|---|
| Logistic Regression | L2, `C=1.0`, `max_iter=2000` |
| Decision Tree | `max_depth=6`, `min_samples_leaf=50` (pruned to limit overfitting) |
| K-Nearest Neighbours | `k=25`, distance-weighted |
| Naive Bayes | `GaussianNB` |
| Support Vector Machine | RBF kernel, `C=1.0`, `gamma="scale"`; AUC computed from `decision_function` |

### Comparison table (held-out test set, 8,238 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8978 | **0.7929** | **0.6514** | 0.1994 | 0.3053 | 0.3220 |
| Decision Tree | 0.8980 | 0.7833 | 0.6202 | 0.2446 | 0.3509 | 0.3462 |
| KNN | 0.8937 | 0.7458 | 0.5622 | 0.2532 | 0.3492 | 0.3287 |
| Naive Bayes | 0.7972 | 0.7711 | 0.3025 | **0.6131** | **0.4051** | 0.3266 |
| Support Vector Machine | **0.8999** | 0.6898 | 0.6501 | 0.2403 | 0.3509 | **0.3544** |

Bold = best value in the column. Precision, Recall and F1 refer to the
positive class (`y = 1`, subscribed). All predictions use the default
decision threshold.

### Observations on model performance

**Context:** a model that always predicts "no" already scores **0.887
accuracy**, so accuracy alone says little here. MCC, F1 and AUC are the more
informative metrics for this imbalanced problem.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest AUC (0.793) and highest precision (0.651), so its probability ranking of clients is the best of the five. At the default 0.5 threshold it is very conservative: it finds only 20 % of real subscribers (recall 0.199), which gives the lowest F1 and MCC. Because the ranking is good, lowering the threshold to 0.2 (the app's threshold slider) raises recall to 0.551 at a precision of 0.422. It is also fast, stable and interpretable. |
| Decision Tree | Second-best MCC (0.346) and accuracy (0.898) with a shallow, pruned tree (depth 6). It picks up non-linear cut-offs in the macro-economic features (`nr_employed`, `euribor3m`) and in `pdays`/`poutcome` that a linear model can only approximate. Its AUC (0.783) is slightly below Logistic Regression because a depth-6 tree outputs only a few distinct leaf probabilities, which makes its ranking coarse. |
| KNN | Weakest ranking model (AUC 0.746). After one-hot encoding there are about 60 dimensions, mostly binary, so Euclidean distances are less meaningful and many clients share identical feature vectors. Recall (0.253) and F1 (0.349) are reasonable, but it gives no gain over the tree. It also has to store the full training set and search it at prediction time. |
| Naive Bayes | Lowest accuracy (0.797) but by far the highest recall (0.613) and the best F1 (0.405). Its independence assumption is clearly violated: `emp_var_rate`, `euribor3m` and `nr_employed` are strongly correlated (pairwise r = 0.91 to 0.97), and the Gaussian likelihood does not suit 0/1 dummy columns. As a result it is over-confident and flags many clients as likely subscribers, trading precision (0.303) for recall. Its MCC (0.327) stays mid-pack. This is useful when missing a potential subscriber costs more than an extra call. |
| Support Vector Machine | Best accuracy (0.900) and best MCC (0.354), with precision (0.650) close to Logistic Regression and better recall (0.240). However, its AUC (0.690) is the lowest: the RBF decision-function values separate the classes well near the boundary but do not rank clients well far from it. It is also by far the slowest model (~90 s to train versus under 1 s for the others, and ~7 s to score the test set) and gives no probabilities without extra calibration. |
| **Overall winner for this dataset** | On the balanced headline metric **MCC, the SVM wins (0.354), with the Decision Tree a close second (0.346)**. The gaps are small (all MCCs fall between 0.32 and 0.35), which shows the task is hard without call duration. For actual campaign targeting, where clients are ranked and the top-N are called, **Logistic Regression** is the most practical choice because of its best AUC, speed and interpretability. If maximising the number of subscribers reached matters most, Naive Bayes's high recall is attractive. |

## Streamlit app features

* **CSV upload** of test data (or one-click use of the bundled `test_data.csv`, which can also be downloaded from the sidebar). Targets may be `0/1` or `yes/no`.
* **Model selection dropdown** covering all five saved models.
* **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1 and MCC shown as metric cards.
* **Confusion matrix**, **ROC curve** and full **classification report**.
* **Compare all models** tab: metrics table with the best value highlighted, plus a bar chart for any chosen metric.
* Optional **custom decision threshold** for probability-based models.
* Download of predictions as CSV.

The app only loads the pre-trained pipelines from `model/` and scores the
uploaded data. It never retrains a model.
