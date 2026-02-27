import numpy as np

from src.data import load_and_split

def test_split_shapes_and_stratification():
    bundle = load_and_split(seed=42)

    # Check non-empty splits
    assert len(bundle.X_train) > 0
    assert len(bundle.X_val) > 0
    assert len(bundle.X_test) > 0

    # Check labels lengths match features
    assert len(bundle.X_train) == len(bundle.y_train)
    assert len(bundle.X_val) == len(bundle.y_val)
    assert len(bundle.X_test) == len(bundle.y_test)

    # Check stratification: class ratios should be similar across splits
    def pos_rate(y):
        y = np.asarray(y)
        return float((y == 1).mean())

    r_train = pos_rate(bundle.y_train)
    r_val = pos_rate(bundle.y_val)
    r_test = pos_rate(bundle.y_test)

    assert abs(r_train - r_val) < 0.05
    assert abs(r_train - r_test) < 0.05

def test_scaling_applied():
    bundle = load_and_split(seed=42)

    # StandardScaler should make train features approximately zero-mean
    train_means = bundle.X_train.mean(axis=0).to_numpy()
    assert np.all(np.abs(train_means) < 1e-6)