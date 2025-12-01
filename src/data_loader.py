import pandas as pd
import pandas_datareader.data as web
from pandas_datareader import data as pdr
import datetime as dt
from .config import ModelParams


def compute_start_date_regression(params: ModelParams) -> dt.date:
    return params.end_date - dt.timedelta(days=365 * params.lookback_years_regression)

def compute_start_date_diagnostics(params: ModelParams) -> dt.date:
    return params.end_date - dt.timedelta(days=365 * params.lookback_years_diagnostics)


def download_market_data_stooq(params: ModelParams, for_diagnostics: bool = False):
    """
    Download daily OHLCV data for key market series from Stooq via pandas_datareader.
    We will use the 'Close' column as the price proxy (similar to Adj Close).

    Stooq tickers:
    - '^SPX'  : S&P 500 index
    - 'HYG.US': HYG ETF (US listing)
    - 'LQD.US': LQD ETF (US listing)
    - 'VXX.US'  : VXX
    """
    
    start_date = (
        compute_start_date_diagnostics(params)
        if for_diagnostics
        else compute_start_date_regression(params)
    )
    end_date = params.end_date

    # Map our logical names to Stooq tickers
    stooq_tickers = {
        "SPX": "^SPX",
        "HYG": "HYG.US",
        "LQD": "LQD.US",
        "VXX": "VXX.US",
    }

    frames = []
    for name, ticker in stooq_tickers.items():
        df_t = web.DataReader(ticker, "stooq", start_date, end_date)
        # Stooq returns most recent first; sort ascending by date
        df_t = df_t.sort_index()
        print(name, ticker, df_t.columns)
        # Keep Close column only and rename to logical name
        df_t = df_t[["Close"]].rename(columns={"Close": name})
        frames.append(df_t)

    market_df = pd.concat(frames, axis=1)

    # Rename columns to match what the rest of the code expects
    market_df.rename(
        columns={
            "SPX": params.sp500_symbol,      # "^GSPC" in your params
            "HYG": params.hyg_symbol,        # "HYG"
            "LQD": params.lqd_symbol,        # "LQD"
            "VXX": params.vxx_symbol,        # "VXX.US"
        },
        inplace=True,
    )

    return market_df

def download_fred_data(params: ModelParams, for_diagnostics: bool = False):
    """
    Download macro series from FRED (global free data).
    """
    
    start_date = (
        compute_start_date_diagnostics(params)
        if for_diagnostics
        else compute_start_date_regression(params)
    )
    fred_ids = list(params.fred_series.values())
    fred_df = pdr.get_data_fred(fred_ids, start=start_date, end=params.end_date)
    fred_df = fred_df.sort_index()

    # Rename columns to the human-readable keys defined in params
    fred_df.rename(columns={v: k for k, v in params.fred_series.items()}, inplace=True)
    return fred_df


def merge_market_macro(market_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge market and macro data on date index, forward-filling low-frequency macro data.
    """
    df = market_df.join(macro_df, how="inner")
    # Forward-fill macro series (monthly/weekly) to align with daily market data
    df = df.ffill()
    df = df.dropna(how="all")
    return df