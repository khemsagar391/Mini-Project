import numpy as np
from sklearn.ensemble import IsolationForest


def test_far_point_is_more_unusual_than_central_point():
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, size=(2000, 3))
    model = IsolationForest(n_estimators=100, random_state=0).fit(normal)
    central = np.zeros((1, 3))
    far = np.full((1, 3), 10.0)
    unusual_central = -model.score_samples(central)[0]
    unusual_far = -model.score_samples(far)[0]
    assert unusual_far > unusual_central


def test_threshold_from_normal_gives_target_false_alarm_rate():
    rng = np.random.default_rng(1)
    normal_train = rng.normal(0, 1, size=(5000, 3))
    normal_val = rng.normal(0, 1, size=(20000, 3))
    model = IsolationForest(n_estimators=100, random_state=1).fit(normal_train)
    unusual_val = -model.score_samples(normal_val)
    threshold = np.quantile(unusual_val, 0.99)
    rate = float((unusual_val > threshold).mean())
    assert abs(rate - 0.01) < 0.002


def test_model_trained_without_labels_outputs_only_scores():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, size=(500, 3))
    model = IsolationForest(n_estimators=50, random_state=2).fit(X)
    scores = model.score_samples(X)
    assert scores.shape == (500,)
    assert np.isfinite(scores).all()