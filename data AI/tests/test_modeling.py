import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from modeling import encode_target, fit_competition, infer_task_type


def test_task_detection_for_numeric_classes_and_continuous_target():
    assert infer_task_type(pd.Series([0, 1, 0, 1])) == "Classification"
    assert infer_task_type(pd.Series(np.arange(30, dtype=float))) == "Regression"
    assert infer_task_type(pd.Series(["a", "b", "a"])) == "Classification"


def test_classification_pipeline_handles_missing_and_unseen_categories():
    frame = pd.DataFrame({
        "amount": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "kind": ["a", "b", None, "a", "b", "a", "b", "a", "b", "a"],
    })
    target = pd.Series(["yes", "no"] * 5)
    y, encoder = encode_target(target, "Classification")
    x_train, x_test, y_train, y_test = train_test_split(
        frame, y, test_size=0.2, random_state=42, stratify=y
    )
    results, models = fit_competition(x_train, y_train, x_test, y_test, "Classification", "F1-Score")
    assert not results.empty
    model = models[results.iloc[0]["Model"]]
    prediction = model.predict(pd.DataFrame({"amount": [3.5], "kind": ["brand-new"]}))
    assert len(prediction) == 1
    assert encoder is not None


def test_regression_pipeline_metrics_and_prediction():
    frame = pd.DataFrame({"size": np.arange(20, dtype=float), "region": ["north", "south"] * 10})
    target = pd.Series(np.arange(20, dtype=float) * 2 + 3)
    y, encoder = encode_target(target, "Regression")
    x_train, x_test, y_train, y_test = train_test_split(frame, y, test_size=0.2, random_state=42)
    results, models = fit_competition(x_train, y_train, x_test, y_test, "Regression", "R2 Score")
    assert "RMSE" in results.columns
    assert encoder is None
    assert np.isfinite(models[results.iloc[0]["Model"]].predict(x_test)).all()
