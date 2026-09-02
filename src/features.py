"""Feature engineering shared by training and every inference surface."""
import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["INCOME_PER_YEAR_EXPERIENCE"] = out["Income"] / (out["Experience"] + 1)
    out["JOB_STABILITY_RATIO"] = out["CURRENT_JOB_YRS"] / (out["Experience"] + 1)
    out["HOUSE_STABILITY_RATIO"] = out["CURRENT_HOUSE_YRS"] / (out["Age"] + 1)
    out["AGE_STARTED_WORK"] = out["Age"] - out["Experience"]
    out["INCOME_BAND"] = pd.cut(
        out["Income"], bins=[-np.inf, 2.5e6, 5e6, 7.5e6, np.inf],
        labels=["low", "mid", "upper_mid", "high"],
    ).astype(str)
    return out

