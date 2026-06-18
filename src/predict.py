import numpy as np
import joblib


def load_model(model_path):
    return joblib.load(model_path)


def predict_uplift(model, X):
    uplift = model.effect(X)
    return np.array(uplift).reshape(-1)