import pandas as pd


def load_data(path):
    """
    Load the raw Telco Customer Churn dataset.
    """
    return pd.read_csv(path)


def clean_data(df):
    """
    Clean the dataset by:
    - Removing duplicate rows
    - Converting TotalCharges to numeric
    - Filling missing TotalCharges with the median
    - Converting Churn to binary (Yes -> 1, No -> 0)
    """

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Fill missing values with median
    median_value = df["TotalCharges"].median()

    df["TotalCharges"] = (
        df["TotalCharges"]
        .fillna(median_value)
    )

    # Convert Churn to binary
    df["Churn"] = (
        df["Churn"]
        .str.strip()
        .map({
            "Yes": 1,
            "No": 0
        })
    )

    # Verify mapping worked correctly
    if df["Churn"].isnull().any():
        raise ValueError(
            "Unexpected values found in the Churn column."
        )

    return df


def save_processed(df, output_path):
    """
    Save the cleaned dataset.
    """
    df.to_csv(output_path, index=False)