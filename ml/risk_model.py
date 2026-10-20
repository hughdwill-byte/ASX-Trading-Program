"""Trailing, availability-safe covariance for the research optimiser (F14).

Daily simple returns up to and including the decision close are the only input. The estimate is scaled to
the forecast horizon (sessions) so it is in the same units as horizon arithmetic returns, then made
finite and positive semi-definite. Every held name and the core ETF are included even when they are not
purchase-eligible.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

RISK_MODEL_ID = "ledoitwolf_daily_x_horizon_v1"


def nearest_psd(S: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    S = (S + S.T) / 2
    vals, vecs = np.linalg.eigh(S)
    return (vecs * np.maximum(vals, floor)) @ vecs.T


def covariance(returns: pd.DataFrame, names: Sequence[str], horizon: int = 21, min_obs: int = 120,
               var_floor: float = 1e-8) -> Dict[str, object]:
    """Ledoit-Wolf shrunk covariance of trailing daily returns, x horizon.

    returns: rows = sessions up to the decision close (caller guarantees no later rows).
    Missing returns are filled with 0 after demeaning (a conservative, shrink-towards-zero-correlation choice);
    names with fewer than `min_obs` observations get the cross-sectional median variance and zero correlation,
    flagged in `thin`. Constant series get `var_floor`.
    """
    from sklearn.covariance import LedoitWolf

    names = list(names)
    R = returns.reindex(columns=names).astype(float)
    obs = R.notna().sum()
    thin = [n for n in names if obs[n] < min_obs]
    good = [n for n in names if n not in thin]
    S = np.zeros((len(names), len(names)))
    shrink = float("nan")
    if len(good) >= 1 and len(R) >= 2:
        X = R[good]
        X = (X - X.mean()).fillna(0.0).to_numpy()
        lw = LedoitWolf().fit(X)
        pos = [names.index(n) for n in good]
        S[np.ix_(pos, pos)] = lw.covariance_
        shrink = float(lw.shrinkage_)
    diag = np.diag(S).copy()
    med = float(np.median(diag[[names.index(n) for n in good]])) if good else 0.04 ** 2
    for n in thin:
        k = names.index(n)
        S[k, :] = S[:, k] = 0.0
        S[k, k] = med
    for k in range(len(names)):
        if not S[k, k] > var_floor:
            S[k, k] = var_floor
    S = nearest_psd(np.nan_to_num(S) * horizon)
    return {"cov": pd.DataFrame(S, index=names, columns=names), "thin": thin, "shrinkage": shrink,
            "model_id": RISK_MODEL_ID, "horizon": horizon, "observations": int(len(R))}
