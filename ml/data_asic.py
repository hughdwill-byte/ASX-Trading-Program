"""ASIC daily aggregated short positions (free, published ~T+3).

Files: https://download.asic.gov.au/short-selling/RR{YYYYMMDD}-001-SSDailyAggShortPos.csv
(same pattern as update_short_positions_csv.py). Cached to data/asic_short/YYYYMMDD.csv,
1 request per second, resumable. A 404 is remembered in data/asic_short/missing.txt (and skipped on reruns)
only once the report date is older than `retry_days` calendar days: recent 404s are usually "not published
yet" and are retried. Every download is logged in data/asic_short/downloads.csv (report date, downloaded_at).
Usable date = report date + `lag_days` business days (the publication time itself is not available).
"""

from __future__ import annotations

import io
import logging
import pathlib
import time
from typing import Optional

import pandas as pd
import requests

from update_short_positions_csv import DAILY_URL

logger = logging.getLogger(__name__)


def download(base: pathlib.Path, start: str, end: Optional[str] = None, pause: float = 1.0, retry_days: int = 14,
             session=None, today: Optional[pd.Timestamp] = None) -> dict:
    base.mkdir(parents=True, exist_ok=True)
    missing_path = base / "missing.txt"
    missing = set(missing_path.read_text().split()) if missing_path.exists() else set()
    got = failed = 0
    session = session or requests.Session()
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    log = []
    for day in pd.bdate_range(start, end or today):
        tag = f"{day:%Y%m%d}"
        dest = base / f"{tag}.csv"
        if dest.exists() or tag in missing:
            continue
        for attempt in range(3):
            try:
                r = session.get(DAILY_URL.format(date=tag), timeout=30)
                if r.status_code == 404:
                    if (today - day).days > retry_days:
                        missing.add(tag)   # old enough: treat as never published (holiday / no report)
                elif r.ok and len(r.content) > 100:
                    dest.write_bytes(r.content)
                    log.append({"report_date": tag, "downloaded_at": pd.Timestamp.now(tz="UTC").isoformat()})
                    got += 1
                else:
                    r.raise_for_status()
                break
            except Exception as err:  # pragma: no cover - network
                if attempt == 2:
                    failed += 1
                    logger.warning("ASIC %s failed: %s", tag, err)
                time.sleep(2 ** attempt)
        time.sleep(pause)
    missing_path.write_text("\n".join(sorted(missing)))
    if log:
        lp = base / "downloads.csv"
        pd.DataFrame(log).to_csv(lp, mode="a", header=not lp.exists(), index=False)
    logger.info("ASIC download: %d new files, %d failures.", got, failed)
    return {"new": got, "failed": failed}


def _read(path: pathlib.Path) -> pd.DataFrame:
    raw = path.read_bytes()
    for enc, sep in (("utf-8-sig", ","), ("utf-16", "\t"), ("latin-1", ",")):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep)
            if df.shape[1] >= 3:
                return df
        except Exception:
            continue
    return pd.DataFrame()


def load_si(base: pathlib.Path, lag_days: int = 4) -> Optional[pd.DataFrame]:
    """date x ticker frame of short interest (fraction, 0.05 = 5%), keyed by the date it becomes usable."""
    rows = {}
    for path in sorted(base.glob("*.csv")):
        df = _read(path)
        if df.empty:
            continue
        cols = {c.lower().strip(): c for c in df.columns}
        code = next((cols[c] for c in cols if "product code" in c or c in ("code", "asx code")), None)
        pct = next((cols[c] for c in cols if "% of total product" in c or "percent" in c), None)
        if code is None or pct is None:
            continue
        values = pd.to_numeric(df[pct].astype(str).str.replace("%", "").str.strip(), errors="coerce") / 100
        usable = pd.Timestamp(path.stem) + pd.offsets.BDay(lag_days)
        rows[usable] = pd.Series(values.to_numpy(), index=df[code].astype(str).str.strip().str.upper())
    if not rows:
        return None
    frame = pd.DataFrame({d: s[~s.index.duplicated()] for d, s in rows.items()}).T.sort_index()
    return frame


def as_of(si: pd.DataFrame, sessions: pd.DatetimeIndex, max_age_days: int = 10) -> pd.DataFrame:
    """As-of join of SI (keyed by usable date, which may fall on a non-session day) onto sessions. The join runs
    on the union of dates BEFORE selecting sessions, so a non-session update is not lost. Values older than
    `max_age_days` calendar days are NaN (stale = missing, never zero)."""
    union = si.index.union(sessions)
    vals = si.reindex(union).ffill().reindex(sessions)
    stamp = pd.Series(si.index, index=si.index).reindex(union).ffill().reindex(sessions)
    fresh = (sessions - pd.DatetimeIndex(stamp)).days <= max_age_days
    return vals.where(pd.Series(fresh, index=sessions), axis=0)


def coverage(base: pathlib.Path, start: str = "2016-01-01") -> dict:
    files = sorted(p.stem for p in base.glob("*.csv"))
    if not files:
        return {"files": 0, "first": None, "last": None, "complete": False}
    expected = len(pd.bdate_range(max(start, f"{files[0][:4]}-{files[0][4:6]}-{files[0][6:]}"), files[-1]))
    first = pd.Timestamp(files[0])
    return {"files": len(files), "first": first.date().isoformat(), "last": pd.Timestamp(files[-1]).date().isoformat(),
            "complete": bool(first <= pd.Timestamp("2016-03-31") and len(files) >= 0.8 * expected)}
