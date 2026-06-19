"""
Download ASIC short-position reports (daily + YTD) and build JSON caches.

Outputs:
  data/short_positions/YYYYMMDD.csv            (daily snapshot when available)
  data/short_positions/YYYYMMDD.json           (parsed daily entries)
  data/short_positions/ytd/YYYYMMDD.csv        (full YTD CSV snapshot)
  data/short_positions/dated/YYYYMMDD.json     (per-day aggregated entries from YTD)
  data/short_positions/latest.json             (most recent entries)
"""

from __future__ import annotations

import datetime as dt
import json
import math
import pathlib
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests

DAILY_URL = "https://download.asic.gov.au/short-selling/RR{date}-001-SSDailyAggShortPos.csv"
YTD_URL = "https://download.asic.gov.au/short-selling/RR{date}-001-SSDailyYTD.csv"
BASE_DIR = pathlib.Path("data") / "short_positions"
DATED_DIR = BASE_DIR / "dated"
YTD_DIR = BASE_DIR / "ytd"


def ensure_directory(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: pathlib.Path) -> bool:
    response = requests.get(url, timeout=30)
    if response.status_code == 404:
        return False
    response.raise_for_status()
    ensure_directory(dest.parent)
    dest.write_bytes(response.content)
    return True


def normalise_columns(df: pd.DataFrame) -> Dict[str, str]:
    return {str(col).lower().strip(): col for col in df.columns}


def _column_parts(column: Any) -> List[str]:
    if isinstance(column, tuple):
        return [
            str(part).strip()
            for part in column
            if part is not None and str(part).strip()
        ]
    return [str(column).strip()]


def _column_matches(column: Any, *keywords: str) -> bool:
    if not keywords:
        return False
    parts = [part.lower() for part in _column_parts(column)]
    combined = " ".join(parts)
    for keyword in keywords:
        candidate = keyword.strip().lower()
        if candidate in parts or combined == candidate:
            return True
    return False


def normalise_number(value: Any, *, round_int: bool = False) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text in {"-", "n/a", "na", "null", "nan"}:
            return None
        text = text.replace(",", "")
        text = text.replace("%", "")
        # Some CSVs use Unicode non-breaking space
        text = text.replace("\u00a0", "")
        try:
            number = float(text)
        except ValueError:
            return None
    else:
        try:
            if pd.isna(value):  # type: ignore[arg-type]
                return None
        except TypeError:
            pass
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

    if math.isnan(number):
        return None
    if round_int:
        return float(int(round(number)))
    return number


def to_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        return text
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except TypeError:
        pass
    return str(value).strip() or None


def parse_entries(
    df: pd.DataFrame,
    fallback_date: Optional[dt.date] = None,
) -> List[Dict[str, Any]]:
    columns = normalise_columns(df)
    symbol_col = (
        columns.get("asx code")
        or columns.get("symbol")
        or columns.get("product code")
        or columns.get("code")
    )
    name_col = (
        columns.get("security name")
        or columns.get("name")
        or columns.get("product")
    )
    percent_col = (
        columns.get("percent of issued capital short sold")
        or columns.get("percent")
        or columns.get("% of total product in issue reported as short positions")
        or columns.get("short percent")
    )
    short_col = (
        columns.get("aggregate gross short position")
        or columns.get("short positions")
        or columns.get("reported short positions")
    )
    total_col = (
        columns.get("total issued")
        or columns.get("total")
        or columns.get("total product in issue")
    )
    date_col = columns.get("reporting date") or columns.get("date")

    entries: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        symbol = str(row.get(symbol_col, "")).strip().upper() if symbol_col else ""
        if not symbol:
            continue
        raw_date = row.get(date_col) if date_col else None
        parsed_date = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(parsed_date) and fallback_date is not None:
            parsed_date = pd.Timestamp(fallback_date)
        date_iso = parsed_date.date().isoformat() if not pd.isna(parsed_date) else None
        short_val = normalise_number(row.get(short_col), round_int=True) if short_col else None
        if short_val is not None:
            short_val = int(short_val)
        total_val = normalise_number(row.get(total_col), round_int=True) if total_col else None
        if total_val is not None:
            total_val = int(total_val)
        percent_val = normalise_number(row.get(percent_col)) if percent_col else None
        entry = {
            "rank": int(row.get("rank")) if "rank" in df.columns else idx + 1,
            "symbol": symbol,
            "name": to_optional_str(row.get(name_col)) if name_col else None,
            "short_positions": short_val,
            "total": total_val,
            "float_total": total_val,
            "percent": percent_val,
            "date": date_iso,
        }
        entries.append(entry)
    return entries


def write_json(path: pathlib.Path, payload: Dict) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def is_date_label(label: str) -> bool:
    try:
        dt.datetime.strptime(label.strip(), "%d/%m/%Y")
        return True
    except Exception:
        return False


def parse_ytd_wide(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    if not isinstance(df.columns, pd.MultiIndex):
        return {}

    name_col = None
    code_col = None
    for column in df.columns:
        if _column_matches(column, "product", "security name"):
            name_col = column
        if _column_matches(column, "product code", "asx code", "symbol"):
            code_col = column
    if code_col is None:
        return {}

    names = df[name_col] if name_col is not None else [None] * len(df)
    symbols = df[code_col]
    dates: List[str] = []
    for column in df.columns:
        if isinstance(column, tuple) and column and isinstance(column[0], str):
            top_label = column[0].strip()
            if is_date_label(top_label) and top_label not in dates:
                dates.append(top_label)

    entries_by_date: Dict[str, List[Dict[str, Any]]] = {}
    for label in dates:
        try:
            parsed_date = dt.datetime.strptime(label, "%d/%m/%Y").date()
        except Exception:
            continue

        def find_subcolumn(*keywords: str) -> Optional[Any]:
            for column in df.columns:
                if (
                    isinstance(column, tuple)
                    and len(column) >= 2
                    and str(column[0]).strip() == label
                    and _column_matches(column[1], *keywords)
                ):
                    return column
            return None

        short_col = find_subcolumn("reported short positions", "short positions")
        percent_col = find_subcolumn("% of total product in issue reported as short positions", "percent")
        total_col = find_subcolumn("total product in issue", "total")

        entries: List[Dict[str, Any]] = []
        for idx in range(len(df)):
            symbol = to_optional_str(symbols.iloc[idx])  # type: ignore[index]
            if not symbol:
                continue
            symbol = symbol.upper()

            name = to_optional_str(names.iloc[idx]) if name_col is not None else None  # type: ignore[index]
            short_val = normalise_number(df.at[idx, short_col], round_int=True) if short_col is not None else None
            if short_val is not None:
                short_val = int(short_val)
            percent_val = normalise_number(df.at[idx, percent_col]) if percent_col is not None else None
            total_val = normalise_number(df.at[idx, total_col], round_int=True) if total_col is not None else None
            if total_val is not None:
                total_val = int(total_val)

            if short_val is None and percent_val is None:
                continue

            entries.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "short_positions": short_val,
                    "percent": percent_val,
                    "total": total_val,
                    "float_total": total_val,
                    "date": parsed_date.isoformat(),
                }
            )

        if not entries:
            continue

        def sort_key(entry: Dict[str, Any]) -> Any:
            percent_value = entry.get("percent")
            short_value = entry.get("short_positions")
            percent_numeric = float(percent_value) if percent_value is not None else 0.0
            short_numeric = float(short_value) if short_value is not None else 0.0
            return (
                percent_value is None,
                -percent_numeric if percent_value is not None else 0.0,
                -short_numeric if short_value is not None else 0.0,
            )

        entries.sort(key=sort_key)
        for rank, entry in enumerate(entries, start=1):
            entry["rank"] = rank

        entries_by_date[parsed_date.strftime("%Y%m%d")] = entries

    return entries_by_date


def process_daily_report() -> Optional[Dict]:
    today = dt.date.today()
    for offset in range(0, 8):
        day = today - dt.timedelta(days=offset)
        stamp = day.strftime("%Y%m%d")
        csv_path = BASE_DIR / f"{stamp}.csv"
        if not download_file(DAILY_URL.format(date=stamp), csv_path):
            continue
        df = pd.read_csv(csv_path)
        entries = parse_entries(df, fallback_date=day)
        payload = {
            "source": str(csv_path),
            "date": day.isoformat(),
            "generated_at": dt.datetime.now().isoformat(),
            "entries": entries,
        }
        write_json(csv_path.with_suffix(".json"), payload)
        write_json(BASE_DIR / "latest.json", payload)
        print(f"Daily report {stamp}: {len(entries)} entries")
        return payload
    print("No daily short-position CSV found in the last week.")
    return None


def process_ytd_report() -> Optional[str]:
    today = dt.date.today()
    csv_path: Optional[pathlib.Path] = None
    source_date: Optional[dt.date] = None

    for offset in range(0, 8):
        day = today - dt.timedelta(days=offset)
        stamp = day.strftime("%Y%m%d")
        candidate = YTD_DIR / f"{stamp}.csv"
        if download_file(YTD_URL.format(date=stamp), candidate):
            csv_path = candidate
            source_date = day
            break
    if csv_path is None:
        print("No YTD short-position CSV available in the last week.")
        return None

    dated_entries: Dict[str, List[Dict[str, Any]]] = {}
    assert csv_path is not None

    # Attempt to parse new (wide) ASIC format first.
    try:
        df_multi = pd.read_csv(csv_path, header=[0, 1])
    except ValueError:
        df_multi = None

    if df_multi is not None:
        dated_entries = parse_ytd_wide(df_multi)

    if not dated_entries:
        # Fallback to legacy row-based format.
        df = pd.read_csv(csv_path)
        date_col = normalise_columns(df).get("reporting date") or normalise_columns(df).get("date")
        if not date_col:
            print("YTD CSV missing reporting date column; skipping.")
            return None

        df["__parsed_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
        for date_value, group in df.groupby("__parsed_date"):
            if pd.isna(date_value):
                continue
            entries = parse_entries(group, fallback_date=date_value)
            date_str = date_value.strftime("%Y%m%d")
            payload = {
                "source": str(csv_path),
                "date": date_value.isoformat(),
                "generated_at": dt.datetime.now().isoformat(),
                "entries": entries,
            }
            write_json(DATED_DIR / f"{date_str}.json", payload)
            dated_entries[date_str] = entries
    else:
        for date_str, entries in dated_entries.items():
            date_value = dt.datetime.strptime(date_str, "%Y%m%d").date()
            payload = {
                "source": str(csv_path),
                "date": date_value.isoformat(),
                "generated_at": dt.datetime.now().isoformat(),
                "entries": entries,
            }
            write_json(DATED_DIR / f"{date_str}.json", payload)

    if dated_entries:
        latest_date = max(dated_entries.keys())
        payload = {
            "source": str(csv_path),
            "date": dt.datetime.strptime(latest_date, "%Y%m%d").date().isoformat(),
            "generated_at": dt.datetime.now().isoformat(),
            "entries": dated_entries[latest_date],
        }
        write_json(BASE_DIR / "latest.json", payload)
        if source_date:
            print(
                f"YTD report processed for {len(dated_entries)} dates "
                f"(latest {latest_date}, source {source_date.isoformat()})."
            )
        else:
            print(f"YTD report processed for {len(dated_entries)} dates (latest {latest_date}).")
        return latest_date

    print("YTD CSV contained no dated entries.")
    return None


def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    DATED_DIR.mkdir(parents=True, exist_ok=True)
    YTD_DIR.mkdir(parents=True, exist_ok=True)

    daily_payload = process_daily_report()
    latest_ytd = process_ytd_report()

    if daily_payload and daily_payload.get("entries"):
        print(f"Latest daily report date: {daily_payload.get('date')}")
    elif latest_ytd:
        print(f"Using YTD latest date {latest_ytd} as fallback for latest.json.")
    else:
        print("No short-position data found; latest.json may be empty.")


if __name__ == "__main__":
    main()
