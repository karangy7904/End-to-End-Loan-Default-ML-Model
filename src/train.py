"""End-to-end, leakage-safe loan risk classification workflow."""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
try:
    from src.features import engineer_features
except ModuleNotFoundError:  # supports direct `python src/train.py`
    from features import engineer_features

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"
PREDICTIONS = ROOT / "predictions"
TARGET = "Risk_Flag"
ID_COL = "Id"


def ensure_directories() -> None:
    for folder in [REPORTS, FIGURES, MODELS, PREDICTIONS]:
        folder.mkdir(parents=True, exist_ok=True)


def save_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=160, bbox_inches="tight")
    plt.close()


def data_audit(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    audit = {
        "training_shape": list(train.shape), "test_shape": list(test.shape),
        "training_duplicates": int(train.duplicated().sum()),
        "test_duplicates": int(test.duplicated().sum()),
        "training_missing_cells": int(train.isna().sum().sum()),
        "test_missing_cells": int(test.isna().sum().sum()),
        "target_counts": {str(k): int(v) for k, v in train[TARGET].value_counts().items()},
        "target_rate": float(train[TARGET].mean()),
    }
    quality = pd.DataFrame({
        "dtype": train.dtypes.astype(str),
        "missing_count": train.isna().sum(),
        "missing_pct": train.isna().mean(),
        "unique_values": train.nunique(dropna=False),
    })
    quality.to_csv(REPORTS / "data_quality.csv")

    numeric = train.select_dtypes(include=np.number).drop(columns=[TARGET, ID_COL], errors="ignore")
    rows = []
    for col in numeric:
        q1, q3 = numeric[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({"feature": col, "q1": q1, "q3": q3, "iqr": iqr,
                     "lower_bound": lo, "upper_bound": hi,
                     "outlier_count": int(((numeric[col] < lo) | (numeric[col] > hi)).sum()),
                     "outlier_pct": float(((numeric[col] < lo) | (numeric[col] > hi)).mean())})
    pd.DataFrame(rows).to_csv(REPORTS / "outlier_analysis.csv", index=False)
    (REPORTS / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def create_eda(train: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", palette="deep")
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=train, x=TARGET)
    ax.set(title="Loan risk class distribution", xlabel="Risk flag", ylabel="Applicants")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x()+p.get_width()/2, p.get_height()),
                    ha="center", va="bottom")
    save_figure("target_distribution.png")

    nums = [c for c in train.select_dtypes(include=np.number).columns if c not in [TARGET, ID_COL]]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for col, ax in zip(nums, axes.flat):
        sns.histplot(data=train.sample(min(50000, len(train)), random_state=SEED), x=col,
                     hue=TARGET, bins=30, stat="density", common_norm=False, ax=ax, element="step")
        ax.set_title(col)
    save_figure("numerical_distributions.png")

    corr = train[nums + [TARGET]].corr(numeric_only=True)
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Numerical correlation matrix")
    save_figure("correlation_heatmap.png")

    cats = ["Married/Single", "House_Ownership", "Car_Ownership", "STATE"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for col, ax in zip(cats, axes.flat):
        rates = train.groupby(col, observed=True)[TARGET].agg(["mean", "size"]).sort_values("mean", ascending=False)
        rates.head(15).sort_values("mean").plot.barh(y="mean", legend=False, ax=ax)
        ax.set(title=f"Default rate by {col}", xlabel="Risk rate", ylabel="")
    save_figure("categorical_risk_rates.png")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5))])
    return ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])


def probability_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raw = model.decision_function(X)
    return 1 / (1 + np.exp(-raw))


def metric_row(name: str, y: pd.Series, prob: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    return {"model": name, "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, prob), "pr_auc": average_precision_score(y, prob)}


def benchmark_models(X_train, y_train, X_valid, y_valid, preprocessor) -> pd.DataFrame:
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=4),
        "SVC": SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True,
                   cache_size=1500, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=20,
                                                  class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=180, max_depth=18,
                                                  min_samples_leaf=5, class_weight="balanced_subsample",
                                                  n_jobs=4, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=140, learning_rate=.06,
                                                          max_depth=3, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=220, max_depth=6, learning_rate=.07,
                                   subsample=.85, colsample_bytree=.85, scale_pos_weight=ratio,
                                   eval_metric="logloss", n_jobs=4, random_state=SEED),
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    rng = np.random.default_rng(SEED)
    max_benchmark = 60000
    sample_idx = rng.choice(len(X_train), min(max_benchmark, len(X_train)), replace=False)
    Xb, yb = X_train.iloc[sample_idx], y_train.iloc[sample_idx]
    rows = []
    for name, estimator in models.items():
        # RBF SVC is intentionally benchmarked on a smaller stratified sample.
        if name == "SVC":
            _, Xfit, _, yfit = train_test_split(Xb, yb, test_size=min(20000, len(Xb)),
                                                 stratify=yb, random_state=SEED)
        else:
            Xfit, yfit = Xb, yb
        pipe = Pipeline([("preprocess", clone(preprocessor)), ("model", estimator)])
        scores = cross_validate(pipe, Xfit, yfit, cv=cv,
                                scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"},
                                n_jobs=1)
        pipe.fit(Xfit, yfit)
        prob = probability_scores(pipe, X_valid)
        row = metric_row(name, y_valid, prob)
        row.update({"cv_roc_auc_mean": scores["test_roc_auc"].mean(),
                    "cv_pr_auc_mean": scores["test_pr_auc"].mean(),
                    "cv_f1_mean": scores["test_f1"].mean(),
                    "benchmark_rows": len(Xfit)})
        rows.append(row)
        print(f"Benchmarked {name}: PR-AUC={row['pr_auc']:.4f}")
    result = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    result.to_csv(REPORTS / "model_comparison.csv", index=False)
    return result


def tune_xgboost(X_train, y_train, preprocessor) -> tuple[Pipeline, dict]:
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(eval_metric="logloss", n_jobs=4, random_state=SEED,
                          scale_pos_weight=ratio)
    pipe = Pipeline([("preprocess", clone(preprocessor)), ("model", model)])
    params = {
        "model__n_estimators": randint(180, 421), "model__max_depth": randint(3, 9),
        "model__learning_rate": uniform(.03, .12), "model__min_child_weight": randint(1, 9),
        "model__subsample": uniform(.72, .25), "model__colsample_bytree": uniform(.72, .25),
        "model__reg_lambda": uniform(.5, 4.5), "model__gamma": uniform(0, .8),
    }
    # Tuning on a fixed stratified subset controls runtime while final fit uses all training rows.
    _, Xt, _, yt = train_test_split(X_train, y_train, test_size=min(70000, len(X_train)),
                                     stratify=y_train, random_state=SEED)
    search = RandomizedSearchCV(pipe, params, n_iter=8, scoring="average_precision",
                                cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                                n_jobs=1, verbose=1, random_state=SEED, refit=True)
    search.fit(Xt, yt)
    best = search.best_params_
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(REPORTS / "tuning_results.csv", index=False)
    final = clone(pipe).set_params(**best).fit(X_train, y_train)
    return final, {"best_cv_pr_auc": float(search.best_score_), "best_params": best,
                   "tuning_rows": len(Xt), "iterations": 8}


def optimize_threshold(y_true, prob) -> tuple[float, pd.DataFrame]:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    table = pd.DataFrame({"threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1]})
    table["f1"] = 2 * table.precision * table.recall / (table.precision + table.recall + 1e-12)
    # Select maximum F1; a business-specific cost function can replace this objective.
    best = table.loc[table.f1.idxmax()]
    table.to_csv(REPORTS / "threshold_analysis.csv", index=False)
    return float(best.threshold), table


def explain_model(pipe: Pipeline, X_valid: pd.DataFrame) -> None:
    sample = X_valid.sample(min(1200, len(X_valid)), random_state=SEED)
    prep = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    transformed = prep.transform(sample)
    names = prep.get_feature_names_out()
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        values = values[-1]
    importance = pd.DataFrame({"feature": names,
                               "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(REPORTS / "shap_feature_importance.csv", index=False)
    shap.summary_plot(values, transformed, feature_names=names, show=False, max_display=20)
    save_figure("shap_summary.png")


def evaluation_plots(y, prob, threshold) -> None:
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y, pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=",", cmap="Blues", cbar=False)
    plt.title(f"Confusion matrix (threshold={threshold:.3f})")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    save_figure("confusion_matrix.png")
    fpr, tpr, _ = roc_curve(y, prob)
    precision, recall, _ = precision_recall_curve(y, prob)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(fpr, tpr, label=f"AUC={roc_auc_score(y, prob):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray"); axes[0].legend(); axes[0].set_title("ROC curve")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate")
    axes[1].plot(recall, precision, label=f"AP={average_precision_score(y, prob):.3f}")
    axes[1].axhline(y.mean(), ls="--", color="gray", label="baseline"); axes[1].legend(); axes[1].set_title("Precision-recall curve")
    axes[1].set(xlabel="Recall", ylabel="Precision")
    save_figure("roc_pr_curves.png")


def business_report(audit: dict, metrics: dict, threshold: float, importance: pd.DataFrame) -> None:
    top = ", ".join(importance.head(8).feature.str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False))
    text = f"""# Business recommendations

## Executive summary

The observed risky-applicant rate is {audit['target_rate']:.1%}. The final model reaches ROC-AUC
{metrics['roc_auc']:.3f} and PR-AUC {metrics['pr_auc']:.3f} on an untouched validation set.
At the F1-optimized threshold of {threshold:.3f}, recall is {metrics['recall']:.1%} and precision is
{metrics['precision']:.1%}. This threshold is a demonstration, not a lending policy.

## Recommended operating model

1. Use probability bands instead of a single automatic approval rule: low risk for streamlined
   review, medium risk for document/manual review, and high risk for enhanced verification.
2. Choose the production threshold using the actual cost of a missed risky applicant versus the
   cost and customer impact of a false alert. Re-run `threshold_analysis.csv` under that cost matrix.
3. The strongest model signals include {top}. Treat SHAP values as predictive associations, not
   causal reasons or adverse-action explanations without legal validation.
4. Monitor monthly class rate, score distribution, missing/unknown-category rate, PR-AUC, recall,
   precision, and group-level error rates. Trigger investigation when drift exceeds agreed limits.
5. Require human review, audit logs, explainability checks, and fair-lending/legal review before
   operational use. Profession, city, state, home ownership, marital status, and related proxies may
   create fairness or regulatory risk.

## Limitations

The dataset has no outcome time window, loan amount, repayment history, bureau variables, or stated
sampling methodology. The random holdout estimates in-sample generalization, not future-time or
cross-region performance. A time-based out-of-time test is required before deployment.
"""
    (REPORTS / "business_recommendations.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_directories()
    train = pd.read_csv(DATA / "Training Data.csv")
    test = pd.read_csv(DATA / "Test Data.csv")
    # The supplied files use `Id` in training and `ID` in test data.
    test_id_col = next((c for c in test.columns if c.lower() == ID_COL.lower()), None)
    if test_id_col is None:
        raise ValueError("Test data must contain an Id/ID column.")
    audit = data_audit(train, test)
    create_eda(train)
    X = engineer_features(train.drop(columns=[TARGET, ID_COL], errors="ignore"))
    y = train[TARGET].astype(int)
    X_test = engineer_features(test.drop(columns=[test_id_col], errors="ignore"))
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=.20, stratify=y, random_state=SEED)
    preprocessor = build_preprocessor(X_train)
    comparison = benchmark_models(X_train, y_train, X_valid, y_valid, preprocessor)
    final_model, tuning = tune_xgboost(X_train, y_train, preprocessor)
    valid_prob = final_model.predict_proba(X_valid)[:, 1]
    threshold, _ = optimize_threshold(y_valid, valid_prob)
    metrics = metric_row("Tuned XGBoost", y_valid, valid_prob, threshold)
    (REPORTS / "final_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pred = (valid_prob >= threshold).astype(int)
    (REPORTS / "classification_report.json").write_text(
        json.dumps(classification_report(y_valid, pred, output_dict=True), indent=2), encoding="utf-8")
    evaluation_plots(y_valid, valid_prob, threshold)
    explain_model(final_model, X_valid)
    importance = pd.read_csv(REPORTS / "shap_feature_importance.csv")
    business_report(audit, metrics, threshold, importance)
    metadata = {"target": TARGET, "id_column": ID_COL, "optimized_threshold": threshold,
                "validation_metrics": metrics, "tuning": tuning,
                "feature_engineering": ["income per experience", "job stability ratio",
                                        "house stability ratio", "age started work", "income band"],
                "random_seed": SEED}
    joblib.dump(final_model, MODELS / "final_model_pipeline.joblib", compress=3)
    (MODELS / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    pd.DataFrame({ID_COL: test[test_id_col], "risk_probability": test_prob,
                  TARGET: test_pred}).to_csv(PREDICTIONS / "test_predictions.csv", index=False)
    summary = {"audit": audit, "top_benchmark": comparison.iloc[0].to_dict(),
               "final": metrics, "threshold": threshold, "tuning": tuning}
    (REPORTS / "run_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
