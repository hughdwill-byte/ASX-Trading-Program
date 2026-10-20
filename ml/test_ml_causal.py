"""Causal ML pipeline checks F07-F10 (synthetic; no network).

Run: python ml/test_ml_causal.py
"""

import pathlib
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ml.dataset_builder import (_augment_with_short_reports, apply_preprocessor, fit_preprocessor,  # noqa: E402
                                make_labels, purged_split)


def frame(days=10):
    idx = pd.bdate_range("2024-01-01", periods=days)
    close = pd.Series(np.arange(10.0, 10.0 + days), index=idx)
    return pd.DataFrame({"open": close - 0.5, "close": close, "adj_close": close}, index=idx)


def test_labels_unknown_terminal_and_cumulative_horizon():
    f = make_labels(frame(), horizon=3)
    assert f["target"].iloc[-3:].isna().all() and f["fwd_return"].iloc[-3:].isna().all()   # unknown, not 0
    # row 0: entry at open of row 1 (10.5), exit at close of row 3 (13): cumulative 3-session traded return
    assert abs(f["fwd_return"].iloc[0] - (13 / 10.5 - 1)) < 1e-12 and f["target"].iloc[0] == 1
    assert f["label_end"].iloc[0] == f.index[3]


def test_external_features_as_of():
    idx = pd.MultiIndex.from_product([["BHP"], pd.bdate_range("2024-01-01", periods=5)], names=["symbol", "date"])
    ds = pd.DataFrame({"x": 1.0}, index=idx)
    stats = {"BHP": {"short_percent": 0.04, "short_rank": 3, "short_positions": 10, "short_report_score": 1.0,
                     "short_report_date": pd.Timestamp("2024-01-03")}}
    out = _augment_with_short_reports(ds, stats)
    assert out["short_percent_cached"].iloc[:2].isna().all() and (out["short_percent_cached"].iloc[2:] == 0.04).all()
    assert out["days_since_short_report"].iloc[:2].isna().all() and out["days_since_short_report"].iloc[-1] == 2


def test_fold_only_preprocessing_and_schema():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
    X.loc[:49, "c"] = np.nan                     # all-missing in the training fold
    train = X.iloc[:50]
    pp = fit_preprocessor(train)
    future = X.copy()
    future.iloc[50:] *= 100                      # later rows change wildly
    assert np.allclose(apply_preprocessor(pp, train), apply_preprocessor(fit_preprocessor(future.iloc[:50]), train))
    assert pp["columns"] == ["a", "b"]           # missing-column rule decided inside the fold
    shuffled = train[["c", "b", "a"]]
    assert np.allclose(apply_preprocessor(pp, shuffled), apply_preprocessor(pp, train))   # fitted column order
    assert np.isfinite(apply_preprocessor(pp, train.drop(columns=["b"]))).all()          # missing col imputed
    try:
        apply_preprocessor({**pp, "schema": "old"}, train)
        raise AssertionError("schema mismatch accepted")
    except ValueError:
        pass


def test_purged_date_grouped_split():
    dates = pd.bdate_range("2024-01-01", periods=20)
    idx = pd.DatetimeIndex(np.repeat(dates, 3))                  # three symbols per date
    label_end = pd.Series(idx).shift(-6).fillna(idx[-1]).to_numpy()   # 2-session labels (3 rows per date)
    train, test, info = purged_split(idx, pd.Series(label_end), test_frac=0.25)
    first = pd.Timestamp(info["first_test_date"])
    for d in dates:                                              # every symbol of a date on one side
        rows = idx == d
        assert train[rows].all() or not train[rows].any()
        assert test[rows].all() or not test[rows].any()
    assert (pd.DatetimeIndex(label_end)[train] < first).all() and info["purged_rows"] > 0


def test_predict_schema_and_payload_precedence():
    from sklearn.linear_model import LogisticRegression

    from ml import predict

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(200, 2)), columns=["f1", "f2"])
    y = (X["f1"] > 0).astype(int)
    pp = fit_preprocessor(X)
    model = LogisticRegression().fit(apply_preprocessor(pp, X), y)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "m.pkl"
        joblib.dump({"model": model, "feature_names": ["f1", "f2"], "preprocessor": {"imputer": None}}, path)
        try:
            predict.predict_next_day({"f1": 1.0, "f2": 0.0}, model_path=path)
            raise AssertionError("legacy artifact accepted")
        except RuntimeError:
            pass
        path = pathlib.Path(tmp) / "m2.pkl"   # artifacts are cached by path
        joblib.dump({"schema": "classifier_v2", "model": model, "feature_names": ["f1", "f2"], "preprocessor": pp,
                     "threshold": 0.5}, path)
        history = pd.DataFrame({"f1": [-3.0], "f2": [0.0]})
        hist_only = predict.predict_next_day(None, history=history, model_path=path)["prob_up"]
        both = predict.predict_next_day({"f1": 3.0}, history=history, model_path=path)["prob_up"]
    assert hist_only < 0.5 < both                                # explicit payload overrides the history row


def test_snapshot_features_never_model_inputs():
    from stocktrade import ModelTrainer

    df = pd.DataFrame({"rsi": [1.0], "pe_ratio": [10.0], "short_interest": [0.1], "target": [1.0], "gap_percent": [0.0]})
    assert ModelTrainer._feature_columns(df) == ["rsi", "gap_percent"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all ML causal checks passed")
