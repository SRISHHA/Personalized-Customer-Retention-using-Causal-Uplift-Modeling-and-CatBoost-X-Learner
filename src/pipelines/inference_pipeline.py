# src/pipelines/inference_pipeline.py

import numpy as np
import pandas as pd
import joblib
from src.utils import load_csv


def load_artifacts(model_path, feature_path):
    model = joblib.load(model_path)
    features = joblib.load(feature_path)
    return model, features


def predict_uplift(df, model, feature_cols):

    target_col = "Outcome"
    treatment_col = "Treatment"
    meta_cols = ["Y0", "Y1"]

    X = df[
        [c for c in df.columns
         if c not in [target_col, treatment_col] + meta_cols]
    ]

    X_encoded = pd.get_dummies(
        X,
        drop_first=True
    )

    X_encoded = X_encoded.reindex(
        columns=feature_cols,
        fill_value=0
    )

    X_encoded = X_encoded.astype(float)

    uplift = model.effect(X_encoded)
    uplift = np.array(uplift).reshape(-1)

    df = df.copy()
    df["uplift_pred"] = uplift

    return df


def segment_customers(df):
    def segment(x):
        if x > 0.2:
            return "Persuadable"
        elif x > 0:
            return "Sleeping Dogs"
        elif x > -0.2:
            return "Sure Things"
        else:
            return "Lost Causes"

    df["segment"] = df["uplift_pred"].apply(segment)
    return df


def run_inference(input_path, model_path, feature_path):
    df = load_csv(input_path)

    model, features = load_artifacts(model_path, feature_path)

    df = predict_uplift(df, model, features)

    df = segment_customers(df)

    return df