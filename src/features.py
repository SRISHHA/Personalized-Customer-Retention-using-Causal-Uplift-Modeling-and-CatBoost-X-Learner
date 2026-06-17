import pandas as pd

def create_features(df):

    df = df.copy()

    df["TenureBucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[0, 1, 2, 3]
    )

    df["FamilyIndicator"] = (
        (df["Partner"] == "Yes") |
        (df["Dependents"] == "Yes")
    ).astype(int)

    support_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport"
    ]

    df["SupportBundle"] = (
        (df[support_cols] == "Yes")
        .sum(axis=1)
    )

    streaming_cols = [
        "StreamingTV",
        "StreamingMovies"
    ]

    df["StreamingBundle"] = (
        (df[streaming_cols] == "Yes")
        .sum(axis=1)
    )

    return df