# project3/ml.py
import os
import numpy as np
import pandas as pd

# Headless matplotlib BEFORE pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression

# Optional GOSDT
try:
    from gosdt import GOSDTClassifier, ThresholdGuessBinarizer
    _GOSDT_OK = True
except Exception:
    _GOSDT_OK = False


def train_tree_baseline(X_dum: pd.DataFrame, y: pd.Series, max_depth=3, img_name="tree.png"):
    """
    Train a sklearn Decision Tree (on one-hot features), compute metrics,
    render and save the tree PNG under MEDIA_ROOT. Returns:
      acc, leaves, img_path, model, (Xtr, Xte, ytr, yte)
    """
    Xtr, Xte, ytr, yte = train_test_split(
        X_dum, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = SkDecisionTreeClassifier(max_depth=int(max_depth), random_state=0)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    leaves = clf.get_n_leaves()

    # Save tree image
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    img_path = os.path.join(settings.MEDIA_ROOT, img_name)
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(
        clf,
        feature_names=X_dum.columns,
        class_names=clf.classes_,
        filled=True,
        ax=ax,
    )
    fig.savefig(img_path, bbox_inches="tight")
    plt.close(fig)

    return acc, leaves, img_path, clf, (Xtr, Xte, ytr, yte)


def train_logistic_with_lambda(X_dum: pd.DataFrame, y: pd.Series, lam=0.10):
    """
    L1 Logistic Regression: higher lambda => stronger sparsity.
    Uses C=1/(lam+eps). Returns:
      acc, used_features, model, (Xtr, Xte, ytr, yte)
    """
    lam = float(lam)
    Xtr, Xte, ytr, yte = train_test_split(
        X_dum, y, test_size=0.2, random_state=42, stratify=y
    )
    C_value = 1.0 / (lam + 1e-6)
    model = LogisticRegression(
        penalty="l1", solver="liblinear", C=C_value, max_iter=2000, multi_class="ovr"
    )
    model.fit(Xtr, ytr)
    acc = accuracy_score(yte, model.predict(Xte))
    used_features = int((np.abs(model.coef_) > 0).any(axis=0).sum())
    return acc, used_features, model, (Xtr, Xte, ytr, yte)


def top_coefficients_by_class(lr_model: LogisticRegression, feature_names, topk=6):
    """
    For each class, return top positive/negative coefficients.
    """
    classes = list(lr_model.classes_)
    coefs = lr_model.coef_
    out = []
    for i, cls in enumerate(classes):
        w = coefs[i]
        order_pos = np.argsort(w)[::-1][:topk]
        order_neg = np.argsort(w)[:topk]
        top_pos = [{"feature": feature_names[j], "weight": float(w[j])} for j in order_pos]
        top_neg = [{"feature": feature_names[j], "weight": float(w[j])} for j in order_neg]
        out.append({"class": str(cls), "top_positive": top_pos, "top_negative": top_neg})
    return out


def train_gosdt_with_lambda(X_raw: pd.DataFrame, y: pd.Series, lam=0.01):
    """
    Train GOSDT (optimal sparse tree). Returns (acc, leaves, error or None).
    If gosdt isn't installed, returns (None, None, 'gosdt not installed').
    """
    if not _GOSDT_OK:
        return None, None, "gosdt not installed"

    print("entered train_gosdt_with_lambda")
    lam = float(lam)
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    print("check one")
    # One-hot categoricals then threshold-guess binarize numerics
    Xtr_num = pd.get_dummies(Xtr_raw, drop_first=False)
    Xte_num = pd.get_dummies(Xte_raw, drop_first=False)
    Xte_num = Xte_num.reindex(columns=Xtr_num.columns, fill_value=0)

    binz = ThresholdGuessBinarizer(
        n_estimators=40, max_depth=1, random_state=2021
    ).set_output(transform="pandas")
    Xtr_b = binz.fit_transform(Xtr_num, ytr)
    Xte_b = binz.transform(Xte_num)

    print("check 2")
    clf = GOSDTClassifier(
        regularization=lam,
        depth_budget=5,
        time_limit=10,
        verbose=False,
    )
    print("check 2.5")
    clf.fit(Xtr_b, ytr)
    print("check 3")
    acc = accuracy_score(yte, clf.predict(Xte_b))

    print("check 4")
    leaves = getattr(clf, "leaves_", None)
    print(leaves)
    if leaves is None:
        try:
            leaves = len(clf.rules_)
        except Exception:
            leaves = np.nan

    return acc, leaves, None
