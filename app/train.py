import argparse
import numpy as np
from app.load import load_csv
from app.model import StandardScaler, LogisticRegression
from app.eval.functions import accuracy
from app.eval.cv import cross_validate, train_val_split, print_cv_results


def print_feature_importance(model, feature_names):
    """
    Print feature importance based on logistic regression coefficients.

    Displays:
    - Raw coefficient (log-odds effect)
    - Odds ratio (exp(coef))
    - Direction of effect
    """

    assert model.w is not None, "Model must be fitted."

    coefs = model.w
    odds_ratios = np.exp(coefs)

    rows = sorted(
        zip(feature_names, coefs, odds_ratios),
        key=lambda x: abs(x[1]),  # sort by absolute log-odds magnitude
        reverse=True,
    )

    print("\nFeature Importance (Logistic Regression)")
    print("-" * 60)
    print(f"{'Feature':<20} {'Coef (log-odds)':>15} {'Odds Ratio':>12}  Effect")
    print("-" * 60)

    for name, coef, odds in rows:
        direction = "↑ increases odds" if coef > 0 else "↓ decreases odds"
        print(f"{name:<20} {coef:>15.4f} {odds:>12.4f}  {direction}")

    print("-" * 60)


def eval_fold(X, y, train_idx, val_idx, lr, iters):
    """
    Train and evaluate a single cross-validation fold.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Binary target vector (0/1).
    train_idx : np.ndarray
        Indices for the training split.
    val_idx : np.ndarray
        Indices for the validation split.
    lr : float
        Learning rate for gradient descent.
    iters : int
        Number of training iterations.

    Returns
    -------
    dict
        Dictionary containing validation metrics:
        - accuracy : classification accuracy
        - loss     : binary cross-entropy loss
        - brier    : Brier score (probability calibration metric)
    """

    # Split data according to fold indices
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Fit scaler on training data only to prevent data leakage
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Train logistic regression model on scaled training data
    model = LogisticRegression(lr=lr, n_iters=iters).fit(X_train_s, y_train)

    # Generate validation predictions
    va_pred = model.predict(X_val_s)
    va_proba = model.predict_proba(X_val_s)

    # Return evaluation metrics for this fold
    return {
        "accuracy": accuracy(y_val, va_pred),
        "loss": model.loss(X_val_s, y_val),
        # Brier score measures probability calibration quality
        "brier": float(np.mean((va_proba - y_val) ** 2)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to Raisin CSV file")
    ap.add_argument("--label-col", default="Class")
    ap.add_argument("--out", default="models/model.npz")

    # Default single-split training params
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--val-frac", type=float, default=0.2)

    # Optional cross-validation
    ap.add_argument("--cv", type=int, default=0, help="If >0, run stratified K-fold CV with K folds")
    ap.add_argument("--grid", default=None, help='Hyperparam grid like "0.05,2000;0.1,2000;0.2,1500"')

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y, feature_names, mapping = load_csv(args.csv, label_col=args.label_col)

    if args.cv and args.cv > 1:
        best, results = cross_validate(X, y, k=args.cv, fn=eval_fold, grid=args.grid, seed=args.seed)

        # Retrain once on ALL data with chosen hyperparams
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        model = LogisticRegression(lr=best["lr"], n_iters=best["iters"]).fit(X_s, y)

        # Save full artifact
        model.save_npz(args.out, scaler, feature_names, mapping)

        print("\nLabel mapping:", mapping)
        print_cv_results(results)
        print(f"\nBest by mean_loss: lr={best['lr']} iters={best['iters']}")
        print_feature_importance(model, feature_names)

        print(f"\nSaved >> {args.out}")
        return

    # get splits
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_frac=args.val_frac, seed=args.seed)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = LogisticRegression(lr=args.lr, n_iters=args.iters).fit(X_train_s, y_train)

    train_acc = accuracy(y_train, model.predict(X_train_s))
    val_acc = accuracy(y_val, model.predict(X_val_s))
    train_loss = model.loss(X_train_s, y_train)
    val_loss = model.loss(X_val_s, y_val)

    print("\nLabel mapping:", mapping)
    print(f"Train loss={train_loss:.4f} acc={train_acc:.4f}")
    print(f"Val loss={val_loss:.4f} acc={val_acc:.4f}")
    print_feature_importance(model, feature_names)

    # save model including deterministic mapping
    model.save_npz(args.out, scaler, feature_names, mapping) 

    print(f"Saved >> {args.out}")


if __name__ == "__main__":
    main()