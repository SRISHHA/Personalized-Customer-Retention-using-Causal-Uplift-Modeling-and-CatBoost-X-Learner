import numpy as np
import pandas as pd


def assign_treatment(
    df: pd.DataFrame,
    seed: int = 42,
    base_rate: float = 0.3
) -> pd.DataFrame:
    

    df = df.copy()

    # keep seed for reproducibility (global numpy)
    np.random.seed(seed)

    treatment_prob = (
        base_rate
        + 0.2 * (df["MonthlyCharges"] > 80).astype(int)
        + 0.2 * (df["SeniorCitizen"] == 1).astype(int)
        + 0.2 * (df["tenure"] < 12).astype(int)
    )

    treatment_prob = np.clip(treatment_prob, 0, 1)

    df["Treatment_Prob"] = treatment_prob
    df["Treatment"] = np.random.binomial(1, treatment_prob)

    return df