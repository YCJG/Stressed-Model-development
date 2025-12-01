from dataclasses import dataclass
import datetime as dt
import numpy as np
from typing import Dict

@dataclass
class ModelParams:
    # Time window
    end_date: dt.date = dt.date.today()
    lookback_years_regression: int = 3   # used for model sample
    lookback_years_diagnostics: int = 7  # used for diagnostics

    # Symbols (Stooq / logical)
    sp500_symbol: str = "^GSPC"       # S&P 500 index (mapped from SPX)
    vxx_symbol: str = "VXX.US"        # mapped from VXX.US
    hyg_symbol: str = "HYG"           # mapped from HYG.US
    lqd_symbol: str = "LQD"           # mapped from LQD.US

    # FRED series IDs
    fred_series: Dict[str, str] = None

    # Rolling windows and lags
    vol_window: int = 21
    macro_lag_days: int = 5

    # Train/test split, CV
    test_size_fraction: float = 0.2
    n_splits: int = 10
    max_iter: int = 10000
    l1_ratios: np.ndarray = np.linspace(0.5, 1.0, 11)
    alphas: np.ndarray = np.logspace(-3, 1, 50)
    top_n_features: int = 10
    top_n_models: int = 10

    def __post_init__(self):
        if self.fred_series is None:
            self.fred_series = {
                "EFFR": "EFFR",      # Effective Fed Funds Rate
                "BAA10Y": "BAA10Y",
            }