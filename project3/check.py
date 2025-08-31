# gosdt_smoketest_numpy.py
# Run: python -X faulthandler gosdt_smoketest_numpy.py
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("TBB_NUM_THREADS","1")

import time, sys, platform, numpy as np, pandas as pd
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gosdt import ThresholdGuessBinarizer, GOSDTClassifier

def main():
    print("Python:", sys.version)
    print("OS:", platform.platform())
    import gosdt
    print("gosdt:", getattr(gosdt, "__version__", "(unknown)"))

    df = load_penguins().dropna()
    y_codes = df["species"].astype("category").cat.codes.to_numpy(dtype=np.int32)
    X_oh = pd.get_dummies(df.drop(columns=["species"]), drop_first=False)

    # Binarize numeric features; force numpy C-contiguous float32
    enc = ThresholdGuessBinarizer(n_estimators=40, max_depth=1, random_state=2021).set_output(transform="pandas")
    Xb_df = enc.fit_transform(X_oh, y_codes)
    Xb = np.ascontiguousarray(Xb_df.to_numpy(dtype=np.float32))

    Xtr, Xte, ytr, yte = train_test_split(Xb, y_codes, test_size=0.2, random_state=42, stratify=y_codes)

    # very conservative params for speed/stability
    lam = max(0.10, 1.0 / len(Xtr))
    clf = GOSDTClassifier(regularization=lam, depth_budget=2, time_limit=3, verbose=True)

    print("Before fit (numpy)")
    t0 = time.time()
    clf.fit(Xtr, ytr)     # <-- if it crashes here, it’s a core solver issue
    print("After fit (%.2fs)" % (time.time() - t0))

    yhat = clf.predict(Xte)
    acc = accuracy_score(yte, yhat)
    leaves = getattr(clf, "leaves_", None)
    if leaves is None:
        try: leaves = len(clf.rules_)
        except: leaves = np.nan
    print(f"ACC={acc:.3f}, LEAVES={leaves}")

if __name__ == "__main__":
    main()
