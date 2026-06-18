
import numpy as np
import pandas as pd


def add_treatment_effect(df: pd.DataFrame) -> pd.DataFrame:
    

    df = df.copy()

    df["TreatmentEffect"] = (
        0.10
        + 0.15 * (df["SeniorCitizen"] == 1).astype(int)
        + 0.12 * (df["MonthlyCharges"] > 80).astype(int)
        + 0.10 * (df["tenure"] < 24).astype(int)
    )

    return df


def add_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Base churn probability + treated probability.
    """

    df = df.copy()

    df["BaseProb"] = (
        0.15
        + 0.15 * (df["MonthlyCharges"] > 80).astype(int)
        + 0.10 * (df["SeniorCitizen"] == 1).astype(int)
        + 0.20 * (df["Contract"] == "Month-to-month")
        - 0.10 * (df["tenure"] > 36).astype(int)
    )

    df["BaseProb"] = np.clip(df["BaseProb"], 0.01, 0.95)

    df["TreatedProb"] = df["BaseProb"] - df["TreatmentEffect"]
    df["TreatedProb"] = np.clip(df["TreatedProb"], 0.01, 0.95)

    return df


import numpy as np
import pandas as pd


def generate_potential_outcomes(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    

    df = df.copy()

    # ensure reproducibility (global numpy style like notebook)
    np.random.seed(seed)

    if "Treatment" not in df.columns:
        raise ValueError("Treatment column missing. Run assign_treatment first.")

    
    df["Y0"] = np.random.binomial(
        1,
        1-df["BaseProb"]
    )

    df["Y1"] = np.random.binomial(
        1,
        1-df["TreatedProb"]
    )

    # observed outcome
    df["Outcome"] = np.where(
        df["Treatment"] == 1,
        df["Y1"],
        df["Y0"]
    )

    return df


def simulate_uplift_dataset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Full pipeline (exact notebook replication).
    """

    df = add_treatment_effect(df)
    df = add_probabilities(df)
    df = generate_potential_outcomes(df, seed=seed)

    return df