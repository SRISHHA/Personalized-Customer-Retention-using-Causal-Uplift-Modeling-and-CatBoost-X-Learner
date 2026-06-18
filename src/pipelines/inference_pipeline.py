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
    X = df[feature_cols]

    uplift = model.effect(X)
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