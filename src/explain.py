import shap
import joblib


def load_explainer(model, X_sample):
    return shap.Explainer(model.predict, X_sample)


def explain(model, X):
    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)
    return shap_values