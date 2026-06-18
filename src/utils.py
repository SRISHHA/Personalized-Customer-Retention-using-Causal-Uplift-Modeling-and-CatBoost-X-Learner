import pandas as pd
import joblib


def load_csv(path):
    return pd.read_csv(path)


def save_csv(df, path):
    df.to_csv(path, index=False)


def load_pickle(path):
    return joblib.load(path)