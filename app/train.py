import argparse
import csv
import numpy as np

from model import StandardScaler, LogisticRegressionScratch


def load_raisin_csv(path: str, label_col: str = "Class"):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("Empty CSV")

    feature_names = [c for c in rows[0].keys() if c != label_col]
    X = np.array([[float(r[c]) for c in feature_names] for r in rows], dtype=np.float64)

    labels = [r[label_col] for r in rows]
    classes = sorted(set(labels))
    if len(classes) != 2:
        raise ValueError(f"Expected 2 classes, got {classes}")

    mapping = {classes[0]: 0, classes[1]: 1}
    y = np.array([mapping[v] for v in labels], dtype=np.int64)

    return X, y, feature_names, mapping


def train_val_split(X, y, val_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    n_val = int(len(idx) * val_frac)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def stratified_kfold_indices(y: np.ndarray, k: int, seed: int):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    folds = [[] for _ in range(k)]
    for i, ix in enumerate(idx0):
        folds[i % k].append(ix)
    for i, ix in enumerate(idx1):
        folds[i % k].append(ix)

    all_idx = np.arange(len(y))
    for i in range(k):
        val_idx = np.array(folds[i], dtype=int)
        tr_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)
        yield tr_idx, val_idx


def eval_fold(X, y, tr_idx, va_idx, lr, iters):
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    model = LogisticRegressionScratch(lr=lr, n_iters=iters).fit(X_tr_s, y_tr)

    va_pred = model.predict(X_va_s)
    va_proba = model.predict_proba(X_va_s)

    return {
        "val_acc": accuracy(y_va, va_pred),
        "val_loss": model.loss(X_va_s, y_va),
        "val_brier": float(np.mean((va_proba - y_va) ** 2)),
    }


def parse_grid(grid_str: str | None):
    # Format: "0.05,2000;0.1,2000;0.2,1500"
    if not grid_str:
        return [
            {"lr": 0.05, "iters": 2000},
            {"lr": 0.1,  "iters": 2000},
            {"lr": 0.2,  "iters": 1500},
        ]
    out = []
    for part in grid_str.split(";"):
        lr_s, it_s = part.split(",")
        out.append({"lr": float(lr_s), "iters": int(it_s)})
    return out


def cross_validate(X, y, k, seed, grid):
    results = []
    for cfg in grid:
        fold_metrics = []
        for tr_idx, va_idx in stratified_kfold_indices(y, k=k, seed=seed):
            fold_metrics.append(eval_fold(X, y, tr_idx, va_idx, cfg["lr"], cfg["iters"]))

        mean_acc = float(np.mean([m["val_acc"] for m in fold_metrics]))
        mean_loss = float(np.mean([m["val_loss"] for m in fold_metrics]))
        mean_brier = float(np.mean([m["val_brier"] for m in fold_metrics]))

        results.append({**cfg, "mean_acc": mean_acc, "mean_loss": mean_loss, "mean_brier": mean_brier})

    best = min(results, key=lambda r: r["mean_loss"])
    return best, results


def print_cv_table(results):
    print("\nCV results:")
    print("  lr     iters   mean_loss   mean_acc   mean_brier")
    for r in results:
        print(f"  {r['lr']:<6g} {r['iters']:<7d} {r['mean_loss']:<10.4f} {r['mean_acc']:<9.4f} {r['mean_brier']:<10.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to Raisin CSV file")
    ap.add_argument("--label-col", default="Class")
    ap.add_argument("--out", default="model.npz")

    # Default single-split training params
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--val-frac", type=float, default=0.2)

    # Optional cross-validation
    ap.add_argument("--cv", type=int, default=0, help="If >0, run stratified K-fold CV with K folds")
    ap.add_argument("--grid", default=None, help='Hyperparam grid like "0.05,2000;0.1,2000;0.2,1500"')

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y, feature_names, mapping = load_raisin_csv(args.csv, label_col=args.label_col)

    if args.cv and args.cv > 1:
        grid = parse_grid(args.grid)
        best, results = cross_validate(X, y, k=args.cv, seed=args.seed, grid=grid)
        print_cv_table(results)
        print(f"\nBest by mean_loss: lr={best['lr']} iters={best['iters']}")

        # Retrain once on ALL data with chosen hyperparams
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        model = LogisticRegressionScratch(lr=best["lr"], n_iters=best["iters"]).fit(X_s, y)

        # Save full artifact
        # If you updated save_npz to include mapping:
        model.save_npz(args.out, scaler, feature_names, mapping)  # <-- preferred
        # If you did NOT update save_npz, use:
        # model.save_npz(args.out, scaler, feature_names)

        print(f"\nSaved -> {args.out}")
        return

    # Default: single train/val split (quick sanity check)
    X_tr, y_tr, X_va, y_va = train_val_split(X, y, val_frac=args.val_frac, seed=args.seed)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    model = LogisticRegressionScratch(lr=args.lr, n_iters=args.iters).fit(X_tr_s, y_tr)

    tr_acc = accuracy(y_tr, model.predict(X_tr_s))
    va_acc = accuracy(y_va, model.predict(X_va_s))
    tr_loss = model.loss(X_tr_s, y_tr)
    va_loss = model.loss(X_va_s, y_va)

    print("Label mapping:", mapping)
    print(f"Train loss={tr_loss:.4f} acc={tr_acc:.4f}")
    print(f"Val   loss={va_loss:.4f} acc={va_acc:.4f}")

    model.save_npz(args.out, scaler, feature_names, mapping)  # <-- preferred
    # model.save_npz(args.out, scaler, feature_names)

    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()



# def load_raisin_csv(path: str, label_col: str = "Class"):
#     # utf-8-sig removes BOM if present (e.g. '\ufeffArea')
#     with open(path, "r", newline="", encoding="utf-8-sig") as f:
#         reader = csv.DictReader(f)
#         rows = list(reader)

#     if not rows:
#         raise ValueError("Empty CSV")

#     # Use header order from the file (DictReader already normalises keys)
#     feature_names = [c for c in rows[0].keys() if c != label_col]

#     X = np.array([[float(r[c]) for c in feature_names] for r in rows], dtype=np.float64)

#     labels = [r[label_col] for r in rows]
#     classes = sorted(set(labels))
#     if len(classes) != 2:
#         raise ValueError(f"Expected 2 classes, got {classes}")

#     mapping = {classes[0]: 0, classes[1]: 1}
#     y = np.array([mapping[v] for v in labels], dtype=np.int64)

#     return X, y, feature_names, mapping


# def train_val_split(X, y, val_frac=0.2, seed=42):
#     rng = np.random.default_rng(seed)
#     idx = np.arange(X.shape[0])
#     rng.shuffle(idx)
#     n_val = int(len(idx) * val_frac)
#     val_idx = idx[:n_val]
#     tr_idx = idx[n_val:]
#     return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


# def accuracy(y_true, y_pred):
#     return float((y_true == y_pred).mean())


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--csv", required=True, help="Path to Raisin CSV file")
#     ap.add_argument("--label-col", default="Class")
#     ap.add_argument("--out", default="model.npz")
#     ap.add_argument("--lr", type=float, default=0.1)
#     ap.add_argument("--iters", type=int, default=2000)
#     ap.add_argument("--val-frac", type=float, default=0.2)
#     ap.add_argument("--seed", type=int, default=42)
#     args = ap.parse_args()

#     X, y, feature_names, mapping = load_raisin_csv(args.csv, label_col=args.label_col)
#     X_tr, y_tr, X_va, y_va = train_val_split(X, y, val_frac=args.val_frac, seed=args.seed)

#     scaler = StandardScaler()
#     X_tr_s = scaler.fit_transform(X_tr)
#     X_va_s = scaler.transform(X_va)

#     model = LogisticRegressionScratch(lr=args.lr, n_iters=args.iters).fit(X_tr_s, y_tr)

#     tr_acc = accuracy(y_tr, model.predict(X_tr_s))
#     va_acc = accuracy(y_va, model.predict(X_va_s))
#     tr_loss = model.loss(X_tr_s, y_tr)
#     va_loss = model.loss(X_va_s, y_va)

#     print("Label mapping:", mapping)
#     print(f"Train loss={tr_loss:.4f} acc={tr_acc:.4f}")
#     print(f"Val loss={va_loss:.4f} acc={va_acc:.4f}")

#     # If your save_npz currently doesn't store mapping, keep it simple:
#     model.save_npz(args.out, scaler, feature_names)
#     print(f"Saved -> {args.out}")


# if __name__ == "__main__":
#     main()


# import csv
# import numpy as np
# from app.model import StandardScaler, LogisticRegressionScratch


# def load_raisin_csv(path: str, label_col: str = "Class"):
#     with open(path, "r", newline="", encoding="utf-8-sig") as f:
#         reader = csv.DictReader(f)
#         rows = list(reader)

#     if not rows:
#         raise ValueError("Empty CSV")

#     # Feature columns = all except label
#     feature_names = [c for c in rows[0].keys() if c != label_col]

#     X = np.array([[float(r[c]) for c in feature_names] for r in rows], dtype=np.float64)

#     # Map labels to {0,1} deterministically
#     labels = [r[label_col] for r in rows]
#     classes = sorted(set(labels))
#     if len(classes) != 2:
#         raise ValueError(f"Expected 2 classes, got {classes}")

#     mapping = {classes[0]: 0, classes[1]: 1}
#     y = np.array([mapping[v] for v in labels], dtype=np.int64)

#     return X, y, feature_names, mapping


# def train_val_split(X, y, val_frac=0.2, seed=42):
#     rng = np.random.default_rng(seed)
#     idx = np.arange(X.shape[0])
#     rng.shuffle(idx)
#     n_val = int(len(idx) * val_frac)
#     val_idx = idx[:n_val]
#     tr_idx = idx[n_val:]
#     return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


# def accuracy(y_true, y_pred):
#     return float((y_true == y_pred).mean())


# if __name__ == "__main__":
#     X, y, feature_names, mapping = load_raisin_csv("Raisin_Dataset.csv", label_col="Class")
#     X_tr, y_tr, X_va, y_va = train_val_split(X, y, val_frac=0.2, seed=42)

#     scaler = StandardScaler()
#     X_tr_s = scaler.fit_transform(X_tr)
#     X_va_s = scaler.transform(X_va)

#     model = LogisticRegressionScratch(lr=0.1, n_iters=2000).fit(X_tr_s, y_tr)

#     tr_acc = accuracy(y_tr, model.predict(X_tr_s))
#     va_acc = accuracy(y_va, model.predict(X_va_s))
#     tr_loss = model.loss(X_tr_s, y_tr)
#     va_loss = model.loss(X_va_s, y_va)

#     print("Label mapping:", mapping)
#     print(f"Train loss={tr_loss:.4f} acc={tr_acc:.4f}")
#     print(f"Val   loss={va_loss:.4f} acc={va_acc:.4f}")

#     model.save_npz("model.npz", scaler, feature_names)
#     print("Saved -> model.npz")
