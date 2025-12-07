import sys
from pathlib import Path

# Ensure `src` package is importable when running this helper
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_ingestion.historical_collector import HistoricalDataCollector

if __name__ == '__main__':
    out_dir = Path('data/raw')
    collector = HistoricalDataCollector(years_back=3)
    news_df, metrics_df = collector.save_historical_data(out_dir)
    print(f"Saved historical news ({len(news_df)}) and metrics ({len(metrics_df)}) to {out_dir}")
