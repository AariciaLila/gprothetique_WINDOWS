import logging
from typing import Dict, Tuple
import mlflow
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

mlflow.sklearn.autolog()


def train_model(data_train: pd.DataFrame) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    target_cols = ["after_exam_125_Hz", "after_exam_250_Hz", "after_exam_500_Hz", "after_exam_1000_Hz", "after_exam_2000_Hz", "after_exam_4000_Hz", "after_exam_8000_Hz"]


    target = data_train[target_cols]

    features = data_train.drop(target_cols, axis=1)

    regressor = LinearRegression()

    regressor.fit(features, target)

    model_info = mlflow.sklearn.log_model(
        sk_model = regressor, artifact_path="model"
    )
    print("Model info :", model_info.model_uri)

    return regressor


def evaluate_model(regressor: LinearRegression, test_data: pd.DataFrame) -> Dict[str, float]:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    target_cols = ["after_exam_125_Hz", "after_exam_250_Hz", "after_exam_500_Hz", "after_exam_1000_Hz", "after_exam_2000_Hz", "after_exam_4000_Hz", "after_exam_8000_Hz"]

    target = test_data[target_cols]

    features = test_data.drop(target_cols, axis=1)

    y_pred = regressor.predict(features)

    score = r2_score(target, y_pred)
    mae = mean_absolute_error(target, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
    return {"r2_score": score, "mae": mae}
