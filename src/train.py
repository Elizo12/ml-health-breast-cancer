import argparse
import json
from datetime import datetime
import joblib
from sklearn.metrics import roc_auc_score, f1_score

from data import load_and_split
from model import build_model
from utils import set_seeds, ensure_dir

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--outdir", type=str, default="outputs")
    args = p.parse_args()

    set_seeds(args.seed)
    ensure_dir(args.outdir)

    bundle = load_and_split(seed=args.seed)
    clf = build_model(C=args.C, seed=args.seed)

    clf.fit(bundle.X_train, bundle.y_train)

    val_proba = clf.predict_proba(bundle.X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)

    auroc = roc_auc_score(bundle.y_val, val_proba)
    f1 = f1_score(bundle.y_val, val_pred)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_path = f"{args.outdir}/logreg_C{args.C}_seed{args.seed}_{stamp}.joblib"
    joblib.dump({"model": clf, "scaler": bundle.scaler}, ckpt_path)

    log = {
        "timestamp": stamp,
        "seed": args.seed,
        "C": args.C,
        "val_auroc": float(auroc),
        "val_f1": float(f1),
        "checkpoint": ckpt_path
    }
    log_path = f"{args.outdir}/val_log_{stamp}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"[VAL] AUROC={auroc:.4f} F1={f1:.4f} | ckpt={ckpt_path}")

if __name__ == "__main__":
    main()