import argparse
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from data import load_and_split
from utils import set_seeds

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seeds(args.seed)
    bundle = load_and_split(seed=args.seed)

    pack = joblib.load(args.ckpt)
    clf = pack["model"]

    proba = clf.predict_proba(bundle.X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auroc = roc_auc_score(bundle.y_test, proba)
    cm = confusion_matrix(bundle.y_test, pred)

    print(f"[TEST] AUROC={auroc:.4f}")
    print("Confusion matrix:\n", cm)
    print(classification_report(bundle.y_test, pred, digits=4))

    # show one concrete failure mode: false positives with highest confidence
    y = bundle.y_test.to_numpy()
    fp_idx = np.where((pred == 1) & (y == 0))[0]
    if len(fp_idx) > 0:
        worst = fp_idx[np.argmax(proba[fp_idx])]
        print("\nExample failure (high-confidence FP):")
        print("index=", int(worst), "proba_malignant=", float(proba[worst]))
        print("Top 5 features (standardized row values):")
        row = bundle.X_test.iloc[worst].sort_values(ascending=False).head(5)
        print(row)
    else:
        print("\nNo false positives found at threshold 0.5.")

if __name__ == "__main__":
    main()