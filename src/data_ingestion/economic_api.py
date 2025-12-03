"""
World Bank Economic Data Helper (simple stub)

Provides lightweight functions to fetch common economic indicators
for Sri Lanka from the World Bank API. Uses simple caching to
save results to the `data/raw/` directory. If network access
fails, returns simulated sample data.
"""
from pathlib import Path
from typing import Optional
import logging
import requests
import pandas as pd
import datetime

logger = logging.getLogger(__name__)

try:
    from config import RAW_DATA_DIR
except Exception:
    # Support running tests / scripts where package import path differs
    from src.config import RAW_DATA_DIR


def _ensure_cache_dir() -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DATA_DIR


def fetch_worldbank_indicator(indicator: str,
                              country: str = "LK",
                              start: Optional[int] = None,
                              end: Optional[int] = None,
                              cache: bool = True) -> pd.DataFrame:
    """Fetch time series for a World Bank indicator.

    Args:
        indicator: World Bank indicator code (e.g. 'NY.GDP.MKTP.CD')
        country: ISO country code (default 'LK')
        start: start year (inclusive)
        end: end year (inclusive)
        cache: whether to cache result to `data/raw/`

    Returns:
        pd.DataFrame with columns ['date', 'value'] (date as int year)
    """
    cache_dir = _ensure_cache_dir()
    cache_file = cache_dir / f"worldbank_{indicator}_{country}.csv"

    if cache and cache_file.exists():
        try:
            df = pd.read_csv(cache_file)
            logger.info(f"Loaded cached World Bank data from {cache_file}")
            return df
        except Exception:
            logger.warning("Failed to read cache, refetching")

    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
    params = {"format": "json", "per_page": 1000}
    if start and end:
        params["date"] = f"{start}:{end}"

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # World Bank returns [metadata, [records]]
        records = data[1] if isinstance(data, list) and len(data) > 1 else []

        rows = []
        for rec in records:
            year = rec.get("date")
            value = rec.get("value")
            try:
                rows.append({"date": int(year), "value": float(value) if value is not None else None})
            except Exception:
                rows.append({"date": year, "value": value})

        df = pd.DataFrame(rows).sort_values("date")

        if cache:
            try:
                df.to_csv(cache_file, index=False)
                logger.info(f"Cached World Bank data to {cache_file}")
            except Exception:
                logger.warning("Failed to write World Bank cache file")

        return df

    except Exception as e:
        logger.warning(f"World Bank fetch failed: {e}. Returning simulated data.")
        # Fallback: return simulated recent series
        current_year = datetime.datetime.now().year
        years = list(range(current_year - 9, current_year + 1))
        import numpy as np
        values = np.round(np.random.uniform(50, 100, len(years)), 2)
        df = pd.DataFrame({"date": years, "value": values})
        return df


def get_gdp(country: str = "LK", start: Optional[int] = None, end: Optional[int] = None) -> pd.DataFrame:
    """Get GDP (current US$) for a country using World Bank code NY.GDP.MKTP.CD"""
    return fetch_worldbank_indicator("NY.GDP.MKTP.CD", country=country, start=start, end=end)


def get_inflation(country: str = "LK", start: Optional[int] = None, end: Optional[int] = None) -> pd.DataFrame:
    """Get annual inflation (consumer prices, % change) using FP.CPI.TOTL.ZG"""
    return fetch_worldbank_indicator("FP.CPI.TOTL.ZG", country=country, start=start, end=end)


if __name__ == "__main__":
    # Quick manual test
    print("Fetching sample GDP series (may use cached data)...")
    df = get_gdp(start=2015, end=2023)
    print(df.tail())
