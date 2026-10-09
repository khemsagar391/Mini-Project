import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from explain_predictions import if_shap, rf_additivity_error, rf_shap


def _toy_multiclass(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(600, 6))
    y = np.where(X[:, 0] > 0.5, 2, np.where(X[:, 1] > 0.0, 1, 0))
    return X, y


def test_rf_shap_shape_is_flows_by_features_by_classes():
    X, y = _toy_multiclass()
    model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=0).fit(X, y)
    values, expected = rf_shap(model, X[:50])
    assert values.shape == (50, 6, 3)
    assert expected.shape == (3,)


def test_rf_shap_is_additive_to_probabilities():
    X, y = _toy_multiclass()
    model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=0).fit(X, y)
    values, expected = rf_shap(model, X[:50])
    assert rf_additivity_error(model, X[:50], values, expected) < 1e-4


def test_isolation_forest_shap_returns_one_row_per_flow():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 6))
    model = IsolationForest(n_estimators=50, random_state=3).fit(X)
    values, expected, method = if_shap(model, X[:10], X[:100])
    assert values.shape == (10, 6)
    assert isinstance(method, str)