# src/pipelines/training_pipeline.py

from src.preprocess import preprocess_data
from src.features import create_features
from src.treatment import assign_treatment
from src.simulate import simulate_uplift_dataset
from src.train import train_xlearner
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd


def run_training_pipeline(input_path):

    df = pd.read_csv(input_path)

    # Step 1: preprocess
    df = preprocess_data(df)

    # Step 2: features
    df = create_features(df)

    # Step 3: treatment
    df = assign_treatment(df)

    # Step 4: simulate outcomes
    df = simulate_uplift_dataset(df)

    # Step 5: split
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    t = df["Treatment"]

    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, t, test_size=0.2, random_state=42
    )

    # Step 6: train
    model = train_xlearner(X_train, y_train, t_train)

    # Step 7: save feature list
    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

    return model