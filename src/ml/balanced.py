from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.utils.class_weight import compute_class_weight


def make_balanced(
    X: np.ndarray, y: np.ndarray, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a 1:1 balanced subset by undersampling the majority class."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return X, y
    k = min(len(pos), len(neg))
    pos_idx = rng.choice(pos, size=k, replace=False)
    neg_idx = rng.choice(neg, size=k, replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(idx)
    return X[idx], y[idx]


def stable_class_weight(y: np.ndarray, classes=(0, 1)) -> Dict[int, float]:
    """
    Compute a *fixed* class_weight dict from the full (pre-balance) labels.
    Use this for warm_start forests to avoid the sklearn warning and to reflect
    the true priors when benign is rare.
    """
    classes = np.array(list(classes), dtype=np.int64)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(classes[0]): float(w[0]), int(classes[1]): float(w[1])}


@dataclass
class ForestGrowConfig:
    total_trees: int = 100
    trees_per_pass: int = 25
    max_depth: Optional[int] = 20
    min_samples_leaf: int = 2
    max_samples_per_tree: int = 1_000_000
    # Use a fixed dict (e.g., {0: w0, 1: w1}) or None to avoid sklearn warm_start warning
    class_weight: Any = None
    n_jobs: int = 1
    random_state: int = 42
    verbose: bool = False


def grow_forest(
    X: np.ndarray,
    y: np.ndarray,
    cfg: ForestGrowConfig,
    sampler: Optional[
        Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray]]
    ] = None,
) -> RandomForestClassifier:
    """
    Grow a RandomForest in passes using warm_start. If `sampler` is provided,
    each pass calls sampler(X, y, pass_id) to feed a fresh (e.g., re-balanced) subset.
    """
    rf = RandomForestClassifier(
        n_estimators=0,  # grow incrementally
        warm_start=True,
        bootstrap=True,
        max_samples=cfg.max_samples_per_tree,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        class_weight=cfg.class_weight,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )
    grown = 0
    pass_id = 0
    while grown < cfg.total_trees:
        add = min(cfg.trees_per_pass, cfg.total_trees - grown)
        rf.n_estimators = grown + add
        Xp, yp = sampler(X, y, pass_id) if sampler else (X, y)
        if cfg.verbose:
            print(
                f"[RF] pass {pass_id:02d}: fitting trees {grown}->{grown+add} "
                f"on {len(yp):,} rows (class_weight={type(cfg.class_weight).__name__})"
            )
        rf.fit(Xp, yp)
        grown += add
        pass_id += 1
    return rf


@dataclass
class IsoGrowConfig:
    total_estimators: int = 200
    trees_per_pass: int = 50
    max_samples: str | int = "auto"
    contamination: str | float = "auto"
    n_jobs: int = 1
    random_state: int = 42
    verbose: bool = False


def grow_iforest(
    X_benign: np.ndarray,
    cfg: IsoGrowConfig,
    sampler: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
) -> IsolationForest:
    """
    Grow an IsolationForest in passes with warm_start. If `sampler` is provided,
    each pass calls sampler(X_benign, pass_id) to feed a fresh benign subset.
    """
    iforest = IsolationForest(
        n_estimators=0,
        warm_start=True,
        max_samples=cfg.max_samples,
        contamination=cfg.contamination,
        bootstrap=False,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )
    grown = 0
    pass_id = 0
    while grown < cfg.total_estimators:
        add = min(cfg.trees_per_pass, cfg.total_estimators - grown)
        iforest.n_estimators = grown + add
        Xp = sampler(X_benign, pass_id) if sampler else X_benign
        if cfg.verbose:
            print(
                f"[IF] pass {pass_id:02d}: fitting trees {grown}->{grown+add} "
                f"on {len(Xp):,} benign rows"
            )
        iforest.fit(Xp)
        grown += add
        pass_id += 1
    return iforest
