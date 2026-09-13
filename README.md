https://loan-default-risk-predictor-olhz.onrender.com/
# End-to-End Loan Risk Classification

A reproducible supervised machine-learning project for predicting `Risk_Flag`
from applicant demographic, employment, income, housing, profession, and location
features. The repository includes raw-data auditing, EDA, outlier analysis,
leakage-safe preprocessing, six-model comparison, imbalance handling, tuning,
threshold optimization, explainability, saved inference artifacts, and test scoring.

> **Responsible-use note:** This is an educational risk-screening project, not a
> production lending decision system. It uses potentially sensitive or proxy
> attributes. Do not use it for real credit decisions without legal, fairness,
> governance, data-quality, and human-review controls.

## Dataset

| Split | Rows | Columns | Target |
|---|---:|---:|---|
| Training | 252,000 | 13 | `Risk_Flag` |
| Test | 28,000 | 12 | unavailable |

The positive class represents 30,996 training records (12.30%), so model selection
emphasizes PR-AUC, recall, precision, and F1 rather than accuracy alone. `Id` is
excluded from training and retained only in the prediction file.

## Workflow

```text
Raw data → audit → EDA/outliers → feature engineering → stratified holdout
→ numerical/categorical pipelines → 3-fold cross-validation → six models
→ imbalance handling → randomized XGBoost tuning → threshold optimization
→ untouched validation evaluation → SHAP → saved pipeline → test predictions
```

Preprocessing is fitted inside each pipeline/CV fold. Numerical values use median
imputation and scaling. Categorical values use most-frequent imputation and one-hot
encoding with unknown-category protection. The source happens to contain no missing
cells, but imputers remain in place for safer inference.

## Models compared

- Logistic Regression
- RBF Support Vector Classifier
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

Class imbalance is handled with balanced class weights for suitable scikit-learn
models and `scale_pos_weight` for XGBoost. The benchmark uses a reproducible,
stratified 60,000-row sample (20,000 for the more expensive RBF SVC); final tuning
uses 70,000 rows and the chosen pipeline is refit on the full 80% training split.

## Results

The tuned XGBoost model materially improved on the untuned benchmark. On the untouched
50,400-row validation set it achieved:

| Metric | Score |
|---|---:|
|  ROC-AUC  |  0.5103  |
|  Average Precision (AP)  |  0.1279  |
|  Accuracy  |  0.2492  |
|  Precision  |  0.1255  |
|  Recall  |  0.8565  |
|  F1-score  |  0.2189  |
|  F1-optimized threshold  |  0.1967  |

Among untuned benchmark models, Random Forest led with validation PR-AUC 0.4470,
followed by XGBoost at 0.4218 and SVC at 0.3775. Full CV and validation metrics are in
`reports/model_comparison.csv`; tuned results are in `reports/final_metrics.json`.
The optimized threshold maximizes validation F1 as a transparent default; production
teams should replace that objective with explicit false-negative and false-positive costs.

Key visual outputs are in `reports/figures/`, while
`reports/business_recommendations.md` translates the measured performance and SHAP
results into operational guidance and limitations.

## Project structure

```text
loan-risk-ml-project/
├── data/                         # supplied training and test CSVs
├── models/                       # fitted pipeline + metadata/threshold
├── notebooks/                    # interactive-use guide
├── predictions/                  # test predictions
├── app/                          # FastAPI schemas, routes, model service
├── reports/
│   ├── figures/                  # EDA, evaluation, and SHAP plots
│   ├── model_comparison.csv
│   ├── threshold_analysis.csv
│   └── business_recommendations.md
├── src/
│   ├── train.py                  # complete reproducible workflow
│   └── predict.py                # batch inference CLI
├── tests/test_smoke.py
├── gradio_app.py                 # interactive browser demo
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

Score another compatible CSV with the saved pipeline:

```bash
python src/predict.py --input "data/Test Data.csv" --output predictions.csv
```

The saved artifact contains feature engineering-independent preprocessing and the
classifier; `predict.py` applies the same engineered features and the threshold
stored in `models/model_metadata.json`.

## API and interactive demo

The tuned model and threshold are included, so both services work immediately after
installing dependencies:

```bash
uvicorn app.main:app --reload
# Swagger documentation: http://localhost:8000/docs

python gradio_app.py
# Interactive demo: http://localhost:7860
```

The API exposes `/health`, strict `/ready`, and `/predict` endpoints. FastAPI,
Gradio, batch scoring, and training all call the same feature-engineering function,
preventing training/serving drift. The API refuses readiness if either the model or
optimized-threshold metadata is missing.

Run both services with Docker:

```bash
docker compose up --build
```

The API is available on port 8000 and Gradio on port 7860.

## Tests

```bash
pytest -q
```

Tests require the included model to load, require `/ready` to succeed, validate an
actual prediction and optimized threshold, and check invalid-category handling. They
do not treat a missing model/HTTP 503 as a passing prediction test.

## Reproducibility and caveats

- Random seed: 42; stratified 80/20 train/validation split; 3-fold stratified CV.
- Package versions are pinned to the environment used for the included model.
- Validation data is never used to fit preprocessing or tune hyperparameters.
- Random splitting cannot detect future-time drift; use an out-of-time validation
  set before deployment.
- SHAP explains model association, not causality, eligibility, or legal justification.
- The raw dataset does not state the performance window, sampling design, loan
  amount, repayment history, or bureau context; conclusions are limited accordingly.

## License

Code is provided under the MIT License. Verify the source dataset's separate license
and attribution requirements before redistribution.
