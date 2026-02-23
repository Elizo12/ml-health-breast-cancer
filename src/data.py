from dataclasses import dataclass
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

@dataclass
class DataBundle:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    scaler: StandardScaler

def load_and_split(test_size=0.2, val_size=0.2, seed=42) -> DataBundle:
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = data.target.copy()

    # cleaning step: remove constant / near-constant columns (if any)
    nunique = X.nunique()
    X = X.loc[:, nunique > 1]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=seed
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return DataBundle(
        X_train=pd.DataFrame(X_train_s, columns=X_train.columns),
        X_val=pd.DataFrame(X_val_s, columns=X_train.columns),
        X_test=pd.DataFrame(X_test_s, columns=X_train.columns),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        scaler=scaler
    )