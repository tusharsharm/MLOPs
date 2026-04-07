"""
Preprocessing: encode labels and categorical features, split data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocess(df: pd.DataFrame):
    """
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    df = df.copy()

    le = LabelEncoder()
    y = le.fit_transform(df["label"])          # AI → 1, Human → 0

    if "content_type" in df.columns:
        df = pd.get_dummies(df, columns=["content_type"], prefix="ct")

    X = df.drop(columns=["label"])

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.fillna(X.mean(numeric_only=True))
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_names
