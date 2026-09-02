# Business recommendations

## Executive summary

The observed risky-applicant rate is 12.3%. The final model reaches ROC-AUC
0.893 and PR-AUC 0.543 on an untouched validation set.
At the F1-optimized threshold of 0.578, recall is 75.0% and precision is
52.4%. This threshold is a demonstration, not a lending policy.

## Recommended operating model

1. Use probability bands instead of a single automatic approval rule: low risk for streamlined
   review, medium risk for document/manual review, and high risk for enhanced verification.
2. Choose the production threshold using the actual cost of a missed risky applicant versus the
   cost and customer impact of a false alert. Re-run `threshold_analysis.csv` under that cost matrix.
3. The strongest model signals include HOUSE_STABILITY_RATIO, Income, INCOME_PER_YEAR_EXPERIENCE, JOB_STABILITY_RATIO, AGE_STARTED_WORK, Experience, Age, CURRENT_JOB_YRS. Treat SHAP values as predictive associations, not
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
