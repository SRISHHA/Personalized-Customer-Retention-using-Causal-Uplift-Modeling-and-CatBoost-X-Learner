import numpy as np
import pandas as pd


def add_treatment_effect(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create heterogeneous treatment effect.
    """

    df = df.copy()

    df["TreatmentEffect"] = (
        0.08
        + 0.12 * (df["SeniorCitizen"] == 1).astype(int)
        + 0.10 * (df["MonthlyCharges"] > 80).astype(int)
        + 0.08 * (df["tenure"] < 24).astype(int)
    )

    return df


def add_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert features into churn probabilities.
    """

    df = df.copy()

    df["BaseProb"] = (
        0.15
        + 0.15 * (df["MonthlyCharges"] > 80).astype(int)
        + 0.10 * (df["SeniorCitizen"] == 1).astype(int)
        + 0.20 * (df["Contract"].str.lower().eq("month-to-month"))
        - 0.10 * (df["tenure"] > 36).astype(int)
    )

    df["BaseProb"] = np.clip(df["BaseProb"], 0.01, 0.95)

    df["TreatedProb"] = np.maximum(df["BaseProb"] - df["TreatmentEffect"],0.01)

    df["TreatedProb"] = np.clip(df["TreatedProb"], 0.01, 0.95)

    return df


def generate_potential_outcomes(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Generate Y0, Y1 and observed outcome.
    """

    df = df.copy()
    rng0 = np.random.default_rng(seed)
    rng1 = np.random.default_rng(seed + 1)

    if "Treatment" not in df.columns:
        raise ValueError("Treatment column missing. Run assign_treatment first.")

    df["Y0"] = rng0.binomial(
    1,
    1 - df["BaseProb"]
    )

    df["Y1"] = rng1.binomial(
        1,
        1 - df["TreatedProb"]
    )

    df["Outcome"] = np.where(
        df["Treatment"] == 1,
        df["Y1"],
        df["Y0"]
    )

    return df


def simulate_uplift_dataset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Full pipeline to create uplift dataset.
    """

    df = add_treatment_effect(df)
    df = add_probabilities(df)
    df = generate_potential_outcomes(df, seed=seed)

    return df