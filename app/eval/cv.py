import numpy as np
from app.utils import assert_has_args


def print_cv_results(results: list[dict]) -> None:
    """
    Print a formatted table for a list of dictionaries.

    Parameters
    ----------
    results : list[dict]
        List of result dictionaries. All dictionaries should
        share the same keys.
    """

    if not results:
        print("(no results)")
        return

    # Use keys from first row (preserve insertion order)
    keys = list(results[0].keys())

    # Determine column widths dynamically
    col_widths = {}
    for k in keys:
        max_content_width = max(
            len(_format_value(r.get(k))) for r in results
        )
        col_widths[k] = max(len(k), max_content_width)

    # Header
    print("\n Cross-Validation Results:")
    header = "  ".join(k.ljust(col_widths[k]) for k in keys)
    print(header)

    # Rows
    for r in results:
        row = "  ".join(
            _format_value(r.get(k)).ljust(col_widths[k])
            for k in keys
        )
        print(row)


def _format_value(v):
    """
    Format values consistently:
    - Floats to 4 decimal places
    - Ints as-is
    - Others converted to string
    """
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, int):
        return str(v)
    return str(v)


def train_val_split(X, y, val_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    n_val = int(len(idx) * val_frac)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


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


def cross_validate(X, y, k, grid, fn, seed=43, agg_fn=np.mean):

    # evaluate custom fn's arguments to enforce conformity
    assert_has_args(fn, ["X", "y", "train_idx", "val_idx"])

    grid = parse_grid(grid)

    results = []
    for cfg in grid:
        fold_metrics = []
        for train_idx, val_idx in stratified_kfold_indices(y, k=k, seed=seed):
            # return one fold output dict
            ret = fn(X, y, train_idx, val_idx, **cfg)
            
            # validate custom fn output
            if not isinstance(ret, dict):
                raise TypeError(f"custom function '{fn.__name__}' must return a dict type")
            if "loss" not in ret:
                raise KeyError(f"'loss' must be returned as one of the key metrics by custom function '{fn.__name__}'")
            
            fold_metrics.append(fn(X, y, train_idx, val_idx, **cfg))
        
        agg_metrics = {
            f"{agg_fn.__name__}_{k}": float(agg_fn([d[k] for d in fold_metrics]))
            for k in fold_metrics[0]
            }

        results.append({**cfg, **agg_metrics})

    # return element whose mean_loss value is the smallest
    best = min(results, key=lambda r: r["mean_loss"])

    return best, results