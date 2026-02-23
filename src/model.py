from sklearn.linear_model import LogisticRegression

def build_model(C: float = 1.0, max_iter: int = 200, seed: int = 42) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        penalty="l2",
        solver="liblinear",
        random_state=seed,
        max_iter=max_iter
    )