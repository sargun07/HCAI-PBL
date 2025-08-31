# project3/cf.py
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _median_absolute_deviation(col):
    col = pd.to_numeric(col, errors="coerce")
    med = np.nanmedian(col)
    mad = np.nanmedian(np.abs(col - med))
    return float(mad if mad and not math.isnan(mad) else 1.0)

def compute_mad_series(df_raw: pd.DataFrame) -> pd.Series:
    """
    MAD per numeric column; categoricals => 1.0
    """
    mad = {}
    for c in df_raw.columns:
        if pd.api.types.is_numeric_dtype(df_raw[c]):
            mad[c] = _median_absolute_deviation(df_raw[c])
        else:
            mad[c] = 1.0
    return pd.Series(mad)

def one_hot_like_train(row_raw: pd.DataFrame, train_columns):
    """
    Convert a 1-row raw dataframe to dummies aligned to training columns.
    """
    row_dum = pd.get_dummies(row_raw, drop_first=False)
    return row_dum.reindex(columns=train_columns, fill_value=0)

def predict_with_lr(lr_model: LogisticRegression, xrow_dum: pd.DataFrame):
    proba = lr_model.predict_proba(xrow_dum)[0]
    pred = lr_model.classes_[np.argmax(proba)]
    return pred, proba

def cf_cost_mad_weighted_L1(x_raw_new, x_raw_orig, mad_series):
    """
    Sum_j |x_j - x0_j| / MAD_j + 1{categorical changed}
    """
    total = 0.0
    for c in x_raw_orig.columns:
        a = x_raw_new[c].iloc[0]
        b = x_raw_orig[c].iloc[0]
        if pd.api.types.is_numeric_dtype(x_raw_orig[c]):
            total += abs(float(a) - float(b)) / float(mad_series[c])
        else:
            total += 0.0 if a == b else 1.0
    return float(total)


def propose_neighbors(x_raw_orig: pd.DataFrame, raw_schema, n_numeric_steps=3, numeric_step_scale=0.5):
    rng = np.random.RandomState(123)
    candidates = []

    # x_raw_orig is a single-row DataFrame
    idx0 = x_raw_orig.index[0]

    for _ in range(300):
        row = x_raw_orig.copy(deep=True)  # deep copy to avoid view/chained issues
        for c, spec in raw_schema.items():
            if spec["type"] == "num":
                mad = float(spec["mad"])
                step = mad * numeric_step_scale
                if step == 0.0:
                    continue
                k = int(rng.randint(1, n_numeric_steps + 1))
                direction = -1.0 if rng.rand() < 0.5 else 1.0
                cur = float(row.loc[idx0, c])
                row.loc[idx0, c] = cur + direction * k * step
            else:
                # categorical flip sometimes
                if rng.rand() < 0.2:
                    values = [v for v in sorted(spec["values"]) if v != row.loc[idx0, c]]
                    if values:
                        row.loc[idx0, c] = rng.choice(values)
        candidates.append(row)

    # a few direct categorical flips
    for c, spec in raw_schema.items():
        if spec["type"] == "cat":
            for v in list(sorted(spec["values"]))[:2]:
                if v != x_raw_orig.loc[idx0, c]:
                    row = x_raw_orig.copy(deep=True)
                    row.loc[idx0, c] = v
                    candidates.append(row)

    return candidates

