import pandas as pd


def load_data(path):
    """
    Load the raw Telco Customer Churn dataset.

    Parameters:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(path)


def clean_data(df):
    """
    Clean the dataset by:
    - Removing duplicate rows
    - Converting TotalCharges to numeric
    - Filling missing TotalCharges with the median
    - Converting Churn to binary (Yes -> 1, No -> 0)

    Parameters:
        df (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """

    print(f"Original dataset shape: {df.shape}")

    # Remove duplicate rows
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicates}")

    df = df.drop_duplicates()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Check missing values created by conversion
    missing_total = df["TotalCharges"].isnull().sum()
    print(f"Missing values in TotalCharges: {missing_total}")

    # Fill missing values with median
    median_value = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_value)

    # Convert Churn to binary
    df["Churn"] = (
        df["Churn"]
        .str.strip()
        .map({"Yes": 1, "No": 0})
    )

    # Verify mapping worked correctly
    if df["Churn"].isnull().any():
        raise ValueError("Unexpected values found in the Churn column.")

    print(f"Cleaned dataset shape: {df.shape}")

    return df


def save_processed(df, output_path="cleaned_telco.csv"):
    """
    Save the cleaned dataset.

    Parameters:
        df (pd.DataFrame): Cleaned dataset.
        output_path (str): Output CSV file path.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")


def main():
    # File paths
    input_path = "/workspaces/Personalized-Customer-Retention-using-Causal-Uplift-Modeling-and-CatBoost-X-Learner/data/raw/Telco-Customer-Churn.csv"
    output_path = "/workspaces/Personalized-Customer-Retention-using-Causal-Uplift-Modeling-and-CatBoost-X-Learner/processed/cleaned_telco.csv"

    # Load raw data
    df = load_data(input_path)

    # Clean data
    df = clean_data(df)

    # Display final missing values
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    # Save processed data
    save_processed(df, output_path)


if __name__ == "__main__":
    main()
