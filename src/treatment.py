import numpy as np
import pandas as pd


def assign_treatment(
    df: pd.DataFrame,
    seed: int = 42,
    base_rate: float = 0.3
) -> pd.DataFrame:
    """
    Assign treatment based on customer risk signals.
    """

    df = df.copy()
    rng = np.random.default_rng(seed)

    treatment_prob = (
        base_rate
        + 0.2 * (df["MonthlyCharges"] > 80).astype(int)
        + 0.2 * (df["SeniorCitizen"] == 1).astype(int)
        + 0.2 * (df["tenure"] < 12).astype(int)
    )

    treatment_prob = np.clip(treatment_prob, 0, 1)

    df["Treatment"] = rng.binomial(
        1,
        treatment_prob
    )

    return df