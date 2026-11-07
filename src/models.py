"""
Step 4: excursion-risk model.

Question the model answers:
    "This fridge is OK right now - will it go outside its safe range in the next 30 minutes?"

We compare 3 things honestly:
    1. rule baseline         (closer to the limit = riskier; no learning)
    2. logistic regression   (simple, explainable)
    3. XGBoost               (stronger, still tabular; falls back to scikit-learn boosting if not installed)

Fridges are split into train / validation / test (60/15/25). A fridge is NEVER in two
groups, so the test score shows how the model behaves on fridges it has never seen.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # xgboost missing or broken on this PC
    HAS_XGB = False


# --------------------------------------------------------------------------
def split_fridges(fridge_ids, seed=config.SEED, pos_ids=None):
    """Split fridge ids into train / val / test groups (a fridge is never in two groups).

    pos_ids = fridges that have at least one excursion. They are spread over all three
    groups, so validation and test each contain real excursions to learn from / be tested on."""
    rng = np.random.RandomState(seed)

    def cut(items):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_tr = int(round(config.SPLIT[0] * n))
        n_va = int(round(config.SPLIT[1] * n))
        if n >= 3:
            n_va = max(1, n_va)
            n_tr = min(max(1, n_tr), n - n_va - 1)
        return items[:n_tr], items[n_tr:n_tr + n_va], items[n_tr + n_va:]

    ids = sorted(fridge_ids)
    if pos_ids:
        pos = [i for i in ids if i in pos_ids]
        neg = [i for i in ids if i not in pos_ids]
        a, b, c = cut(pos)
        d, e, f = cut(neg)
        parts = (a + d, b + e, c + f)
    else:
        parts = cut(ids)
    return {"train": set(parts[0]), "val": set(parts[1]), "test": set(parts[2])}


def assign_split(table: pd.DataFrame, groups: dict) -> pd.Series:
    def which(fid):
        for name, members in groups.items():
            if fid in members:
                return name
        return "train"
    return table["fridge_id"].map(which)


def rule_score(df: pd.DataFrame) -> np.ndarray:
    """Baseline: risk grows as the temperature gets closer to the upper limit."""
    return np.clip(1 - df["margin_hi"].values / 2.0, 0, 1)


def _models(pos_weight: float):
    models = {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced")),
    }
    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.08, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="logloss",
            random_state=config.SEED, n_jobs=2)
    else:
        models["gradient_boosting_sklearn"] = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.08, random_state=config.SEED, class_weight="balanced")
    return models


def best_f1_threshold(y, score):
    """Pick the probability cut-off with the best F1 on the validation set."""
    best_t, best_f = 0.5, -1
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_score(y, score >= t, zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), float(f)
    return best_t


def evaluate(y, score, threshold):
    pred = score >= threshold
    out = {
        "prevalence": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, score)) if len(set(y)) > 1 else None,
        "pr_auc": float(average_precision_score(y, score)) if len(set(y)) > 1 else None,
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    return out


def _log_mlflow(name, params, metrics, root):
    """Optional: track the experiment with MLflow (skipped if not installed)."""
    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///" + (Path(root) / "mlflow.db").resolve().as_posix())
        mlflow.set_experiment("coldsense_excursion_risk")
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params)
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
    except Exception as e:
        print(f"  (MLflow logging skipped: {e})")


# --------------------------------------------------------------------------
def train_and_compare(table: pd.DataFrame, features, cfg, out_dir: Path):
    """Train all models, return (results_df, best_name, fitted_model, threshold, split_map)."""
    pos_ids = set(table.loc[(table["eligible"] == 1) & (table["y"] == 1), "fridge_id"])
    groups = split_fridges(table["fridge_id"].unique(), pos_ids=pos_ids)
    table["split"] = assign_split(table, groups)
    elig = table[table["eligible"] == 1]
    tr, va, te = (elig[elig["split"] == s] for s in ("train", "val", "test"))
    for name, part in (("train", tr), ("val", va), ("test", te)):
        if len(part) == 0 or part["y"].nunique() < 2:
            raise SystemExit(
                f"The {name} split has no excursions (or no rows) with profile '{cfg.profile}'.\n"
                f"Try another limit, e.g.  python scripts/02_run_pipeline.py --profile household"
            )
    print(f"  fridges: train={len(groups['train'])} val={len(groups['val'])} test={len(groups['test'])}")
    print(f"  rows used: train={len(tr)} val={len(va)} test={len(te)}  "
          f"(excursion rate train={tr['y'].mean():.1%}, test={te['y'].mean():.1%})")

    Xtr, ytr = tr[features], tr["y"].values
    pos_weight = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))

    results, fitted = [], {}

    # 1) rule baseline
    s_val, s_te = rule_score(va), rule_score(te)
    thr = best_f1_threshold(va["y"].values, s_val)
    m = evaluate(te["y"].values, s_te, thr)
    results.append({"model": "rule_baseline", **m, "val_pr_auc": average_precision_score(va["y"], s_val)})
    _log_mlflow("rule_baseline", {"type": "rule", "profile": cfg.profile}, m, config.ROOT)

    # 2) learned models
    for name, model in _models(pos_weight).items():
        model.fit(Xtr, ytr)
        s_val = model.predict_proba(va[features])[:, 1]
        s_te = model.predict_proba(te[features])[:, 1]
        thr = best_f1_threshold(va["y"].values, s_val)
        m = evaluate(te["y"].values, s_te, thr)
        results.append({"model": name, **m, "val_pr_auc": average_precision_score(va["y"], s_val)})
        fitted[name] = (model, thr)
        _log_mlflow(name, {"type": name, "profile": cfg.profile, "n_features": len(features)}, m, config.ROOT)
        print(f"  {name:<28} test PR-AUC={m['pr_auc']:.3f}  ROC-AUC={m['roc_auc']:.3f}  "
              f"recall={m['recall']:.2f}  precision={m['precision']:.2f}")

    res = pd.DataFrame(results)
    print(f"  {'rule_baseline':<28} test PR-AUC={res.iloc[0]['pr_auc']:.3f}  ROC-AUC={res.iloc[0]['roc_auc']:.3f}")

    # choose the final model by VALIDATION PR-AUC (never by the test set)
    learned = res[res["model"] != "rule_baseline"].sort_values("val_pr_auc", ascending=False)
    best_name = learned.iloc[0]["model"]
    model, thr = fitted[best_name]
    print(f"  -> final risk model: {best_name} (chosen on validation data)")

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features, "threshold": thr, "profile": cfg.profile,
                 "hi": cfg.hi, "lo": cfg.lo}, out_dir / "risk_model.joblib")
    res.to_csv(out_dir / "model_comparison.csv", index=False)
    _save_importance(model, best_name, features, out_dir)
    return res, best_name, model, thr, groups


def _save_importance(model, name, features, out_dir):
    try:
        if name == "logistic_regression":
            coefs = model[-1].coef_[0]
            imp = pd.DataFrame({"feature": features, "importance": np.abs(coefs), "signed": coefs})
        else:
            imp = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
        imp.sort_values("importance", ascending=False).to_csv(out_dir / "feature_importance.csv", index=False)
    except Exception:
        pass


def score_all(table: pd.DataFrame, model, features) -> np.ndarray:
    """Risk probability for every row. Rows already outside the limit get 1.0."""
    X = table[features]
    prob = model.predict_proba(X)[:, 1]
    prob = np.where(table["p"].isna(), np.nan, prob)
    prob = np.where(table["over_now"] == 1, 1.0, prob)
    return prob
