# project3/views.py
import json
import numpy as np
import pandas as pd
import platform
import multiprocessing as mp
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier

from .ml import (
    train_tree_baseline,
    train_logistic_with_lambda,
    top_coefficients_by_class,
    train_gosdt_with_lambda,
)
from .cf import (
    compute_mad_series,
    one_hot_like_train,
    predict_with_lr,
    cf_cost_mad_weighted_L1,
    propose_neighbors,
)

def _gosdt_worker(X_raw, y, lam, q):
    """Child process: call your existing train_gosdt_with_lambda and return results."""
    try:
        acc, leaves, err = train_gosdt_with_lambda(X_raw, y, lam)
        q.put({"ok": True, "acc": acc, "leaves": leaves, "err": err})
    except Exception as e:
        # If we get here, it was a Python-level error (not a native crash)
        q.put({"ok": False, "err": repr(e)})

def gosdt_metrics_safe(X_raw, y, lam, timeout=20):
    """
    Run GOSDT in an isolated process so native crashes don't kill Django.
    Returns (acc, leaves, err_msg_or_None).
    """
    # On Windows, use 'spawn' to avoid forking-state weirdness
    if platform.system() == "Windows":
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            # start method was already set earlier in the process
            pass

    q = mp.Queue()
    p = mp.Process(target=_gosdt_worker, args=(X_raw, y, lam, q))
    p.start()
    p.join(timeout)

    if p.is_alive():
        # timed out
        p.terminate()
        p.join()
        return None, None, "gosdt timed out"

    # If GOSDT crashed natively, exitcode will be non-zero and queue may be empty
    if p.exitcode != 0:
        return None, None, f"gosdt crashed (exit {p.exitcode})"

    msg = q.get() if not q.empty() else {}
    if msg.get("ok"):
        return msg["acc"], msg["leaves"], msg.get("err")
    else:
        return None, None, msg.get("err", "gosdt failed")

def make_json_safe(obj):
    """Recursively convert NumPy types to Python builtins for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

def _to_int(val, default):
    try:
        return int(val)
    except Exception:
        return default

def _to_float(val, default):
    try:
        return float(val)
    except Exception:
        return default


def index(request):
    """
    Render the HTML template (no training/work here).
    """
    return render(request, "project3/interface.html")


@require_POST
def train_api(request):
    """
    Train all three models using UI sliders and return metrics + assets.
    Body JSON:
      { "dt_max_depth": int, "lr_lambda": float, "gosdt_lambda": float }
    """
    # parse
    try:
        payload_in = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload_in = {}
    print("...entered train_api....")
    dt_max_depth = _to_int(payload_in.get("dt_max_depth"), 3)
    lr_lambda    = _to_float(payload_in.get("lr_lambda"), 0.10)
    gosdt_lambda = _to_float(payload_in.get("gosdt_lambda"), 0.01)

    dt_max_depth = max(1, min(dt_max_depth, 20))
    lr_lambda    = max(0.0, lr_lambda)
    gosdt_lambda = max(0.0, gosdt_lambda)

    # data
    df = load_penguins().dropna()
    y = df["species"]
    X_raw = df.drop(columns=["species"])
    X_dum = pd.get_dummies(X_raw, drop_first=False)

    # Decision Tree
    tree_acc, tree_leaves, img_path, tree_model, _ = train_tree_baseline(
        X_dum, y, max_depth=dt_max_depth, img_name="tree.png"
    )
    tree_img_url = "/media/tree.png"  # served via MEDIA_URL
    print("decision tree done!")
    # Logistic Regression
    lr_acc, lr_used, lr_model, _ = train_logistic_with_lambda(X_dum, y, lam=lr_lambda)
    feature_names = list(X_dum.columns)
    lr_top = top_coefficients_by_class(lr_model, feature_names, topk=6)
    print("logistic regression done!")
    # GOSDT
    gosdt_acc, gosdt_leaves, gosdt_err = gosdt_metrics_safe(X_raw, y, gosdt_lambda, timeout=20)
    print("gosdt done!")
    payload_out = {
        "inputs": {
            "dt_max_depth": dt_max_depth,
            "lr_lambda": lr_lambda,
            "gosdt_lambda": gosdt_lambda,
        },
        "tree": {
            "accuracy": tree_acc,
            "leaves": tree_leaves,
            "image": tree_img_url,
        },
        "logreg": {
            "accuracy": lr_acc,
            "used_features": lr_used,
            "top_coefficients": lr_top,
        },
        "gosdt": {
            "available": (gosdt_err is None) and (gosdt_acc is not None),
            "accuracy": gosdt_acc,
            "leaves": gosdt_leaves,
            "error": gosdt_err,
        },
    }
    print(payload_out)
    return JsonResponse(make_json_safe(payload_out), status=200)


@require_POST
def counterfactuals_api(request):
    """
    Generate counterfactuals using MAD-weighted L1.
    Body JSON:
      { "model": "logreg"|"tree", "lr_lambda": float, "dt_max_depth": int,
        "index": int, "target": str, "k": int }
    """
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    model_choice = str(payload.get("model", "logreg")).lower()
    lr_lambda     = _to_float(payload.get("lr_lambda"), 0.10)
    dt_max_depth  = _to_int(payload.get("dt_max_depth"), 3)
    idx           = _to_int(payload.get("index"), 0)
    target        = payload.get("target", "Adelie")
    topk          = max(1, _to_int(payload.get("k"), 3))

    # data
    df = load_penguins().dropna()
    y = df["species"].astype("category")
    X_raw = df.drop(columns=["species"])
    X_dum = pd.get_dummies(X_raw, drop_first=False)

    mad_series = compute_mad_series(X_raw)

    # consistent split
    Xtr_d, Xte_d, ytr, yte = train_test_split(
        X_dum, y, test_size=0.2, random_state=42, stratify=y
    )
    Xtr_r = X_raw.loc[Xtr_d.index]
    Xte_r = X_raw.loc[Xte_d.index]

    if idx < 0 or idx >= len(Xte_r):
        idx = 0

    x0_raw = Xte_r.iloc[[idx]].copy()
    true_label = str(yte.iloc[idx])

    results = []
    if model_choice == "tree":
        tree = SkDecisionTreeClassifier(max_depth=int(dt_max_depth), random_state=0)
        tree.fit(Xtr_d, ytr)

        raw_schema = {}
        for c in X_raw.columns:
            if pd.api.types.is_numeric_dtype(X_raw[c]):
                raw_schema[c] = {"type": "num", "mad": float(mad_series[c])}
            else:
                raw_schema[c] = {"type": "cat", "values": set(X_raw[c].dropna().unique())}

        cands = propose_neighbors(x0_raw, raw_schema)
        for cand in cands:
            x_cand_dum = one_hot_like_train(cand, Xtr_d.columns)
            pred = tree.predict(x_cand_dum)[0]
            if str(pred) == str(target):
                cost = cf_cost_mad_weighted_L1(cand, x0_raw, mad_series)
                changes = {c: {"from": x0_raw[c].iloc[0], "to": cand[c].iloc[0]}
                           for c in X_raw.columns if cand[c].iloc[0] != x0_raw[c].iloc[0]}
                results.append({"cost": float(cost), "changes": changes})

    else:  # logistic by default
        C_value = 1.0 / (lr_lambda + 1e-6)
        lr = LogisticRegression(
            penalty="l1", solver="liblinear", C=C_value, max_iter=2000, multi_class="ovr"
        )
        lr.fit(Xtr_d, ytr)

        raw_schema = {}
        for c in X_raw.columns:
            if pd.api.types.is_numeric_dtype(X_raw[c]):
                raw_schema[c] = {"type": "num", "mad": float(mad_series[c])}
            else:
                raw_schema[c] = {"type": "cat", "values": set(X_raw[c].dropna().unique())}

        cands = propose_neighbors(x0_raw, raw_schema)
        for cand in cands:
            x_cand_dum = one_hot_like_train(cand, Xtr_d.columns)
            pred, proba = predict_with_lr(lr, x_cand_dum)
            if str(pred) == str(target):
                cost = cf_cost_mad_weighted_L1(cand, x0_raw, mad_series)
                changes = {c: {"from": x0_raw[c].iloc[0], "to": cand[c].iloc[0]}
                           for c in X_raw.columns if cand[c].iloc[0] != x0_raw[c].iloc[0]}
                results.append({
                    "cost": float(cost),
                    "changes": changes,
                    "confidence": float(np.max(proba)),
                })

    results = sorted(results, key=lambda d: d["cost"])[:topk]
    out = {
        "index": idx,
        "true_label": true_label,
        "target": str(target),
        "model": model_choice,
        "k": topk,
        "counterfactuals": results,
    }
    return JsonResponse(make_json_safe(out), status=200)
