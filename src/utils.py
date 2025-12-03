"""
Utility functions for PulseX Sri Lanka
Helper functions used across the project
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import logging
import hashlib

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path = None, log_level: str = "INFO"):
    """Setup logging configuration"""
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "logs"
    
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"pulsex_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Logging initialized. Log file: {log_file}")


def save_json(data: Any, filepath: Path, indent: int = 2):
    """Save data to JSON file"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        logger.info(f"Saved JSON to {filepath}")
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {str(e)}")
        raise


def load_json(filepath: Path) -> Any:
    """Load data from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {str(e)}")
        raise


def save_pickle(data: Any, filepath: Path):
    """Save data using pickle"""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved pickle to {filepath}")
    except Exception as e:
        logger.error(f"Error saving pickle to {filepath}: {str(e)}")
        raise


def load_pickle(filepath: Path) -> Any:
    """Load data from pickle file"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded pickle from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading pickle from {filepath}: {str(e)}")
        raise


def generate_hash(text: str) -> str:
    """Generate MD5 hash of text"""
    return hashlib.md5(text.encode()).hexdigest()


def time_ago(dt: datetime) -> str:
    """
    Convert datetime to human-readable 'time ago' format
    """
    now = datetime.now()
    diff = now - dt
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return f"{int(seconds)}s ago"
    elif seconds < 3600:
        return f"{int(seconds/60)}m ago"
    elif seconds < 86400:
        return f"{int(seconds/3600)}h ago"
    elif seconds < 604800:
        return f"{int(seconds/86400)}d ago"
    else:
        return dt.strftime('%Y-%m-%d')


def format_number(num: float, decimals: int = 2) -> str:
    """Format number with commas and decimals"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.{decimals}f}M"
    elif num >= 1_000:
        return f"{num/1_000:.{decimals}f}K"
    else:
        return f"{num:.{decimals}f}"


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split list into chunks"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def get_date_range(start_date: datetime, end_date: datetime) -> List[datetime]:
    """Generate list of dates between start and end"""
    dates = []
    current = start_date
    
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    
    return dates


def clean_filename(filename: str) -> str:
    """Clean filename to be filesystem-safe"""
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename


def ensure_dir(directory: Path):
    """Ensure directory exists, create if not"""
    directory.mkdir(parents=True, exist_ok=True)


def get_file_size(filepath: Path) -> str:
    """Get human-readable file size"""
    size_bytes = filepath.stat().st_size
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} TB"


def retry_on_failure(func, max_attempts: int = 3, delay: float = 1.0):
    """Retry function on failure"""
    import time
    
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt < max_attempts - 1:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                time.sleep(delay)
            else:
                logger.error(f"All {max_attempts} attempts failed")
                raise


class Timer:
    """Context manager for timing code execution"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        logger.info(f"{self.name} started")
        return self
    
    def __exit__(self, *args):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"{self.name} completed in {elapsed:.2f}s")


class RateLimiter:
    """Simple rate limiter"""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            now = datetime.now()
            
            # Remove old calls outside time window
            self.calls = [call_time for call_time in self.calls 
                         if (now - call_time).total_seconds() < self.time_window]
            
            if len(self.calls) >= self.max_calls:
                sleep_time = self.time_window - (now - self.calls[0]).total_seconds()
                if sleep_time > 0:
                    logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.2f}s")
                    import time
                    time.sleep(sleep_time)
                    self.calls = []
            
            self.calls.append(now)
            return func(*args, **kwargs)
        
        return wrapper


# Testing
if __name__ == "__main__":
    setup_logging()
    
    print("\nTesting utility functions:")
    print("="*60)
    
    # Test time_ago
    past = datetime.now() - timedelta(hours=2, minutes=30)
    print(f"Time ago: {time_ago(past)}")
    
    # Test format_number
    print(f"Format 1500000: {format_number(1500000)}")
    print(f"Format 3500: {format_number(3500)}")
    
    # Test percentage change
    pct = calculate_percentage_change(100, 125)
    print(f"Percentage change (100 to 125): {pct:.1f}%")
    
    # Test Timer
    with Timer("Test operation"):
        import time
        time.sleep(0.5)
    
    print("="*60)