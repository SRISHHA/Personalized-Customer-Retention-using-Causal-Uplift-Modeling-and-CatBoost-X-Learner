# src/train.py

from econml.metalearners import XLearner
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_xlearner(
    X_train,
    y_train,
    t_train,
    model_path="models/xlearner_model.pkl"
):
    """
    Train an X-Learner model and save it.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix.
    y_train : pd.Series
        Outcome variable.
    t_train : pd.Series
        Treatment assignment (0/1).

    Returns
    -------
    model
        Trained X-Learner model.
    """

    # Base models for treated and control groups
    outcome_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )

    propensity_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Create X-Learner
    model = XLearner(
        models=outcome_model,
        propensity_model=propensity_model
    )

    # Train
    model.fit(
        Y=y_train,
        T=t_train,
        X=X_train
    )

    # Save model
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")

    return model
