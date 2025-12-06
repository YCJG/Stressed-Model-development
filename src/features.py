import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import ModelParams

def compute_sp500_vol(df: pd.DataFrame, params: ModelParams) -> pd.Series:
    """
    Compute 21-day rolling realized volatility of S&P 500 daily returns.
    This is the dependent variable (Y): equity dispersion proxy.
    """
    sp = df[params.sp500_symbol]
    returns = sp.pct_change()
    # 21-day rolling standard deviation of daily returns, annualized
    vol = returns.rolling(window=params.vol_window).std() * np.sqrt(252)
    vol = vol.clip(lower=1e-6)  # avoid log(0)
    vol = np.log(vol) # Use log‑vol to stabilize scale, since extreme spikes dominate the loss
    vol.name = "SPX_21D_REALIZED_VOL"
    return vol


def build_feature_frame(df: pd.DataFrame, params: ModelParams) -> pd.DataFrame:
    """
    Build a feature DataFrame with:
    - Y: 21-day realized volatility of S&P 500
    - X: macro/market drivers (transformed and lagged where appropriate)
    """

    # 1) Target variable (Y)
    y = compute_sp500_vol(df, params)
    # Volatility persistence: previous day's vol
    vol_lag = y.shift(21).rename("VOL_LAG") # Volatility persistence (lag of Y) 
    
    # 2) Base daily returns
    sp_ret = df[params.sp500_symbol].pct_change().rename("SP_RET")
    hyg_ret = df[params.hyg_symbol].pct_change().rename("HYG_RET")
    lqd_ret = df[params.lqd_symbol].pct_change().rename("LQD_RET")
    vxx_ret = df[params.vxx_symbol].pct_change().rename("VXX_RET")  
    
    # Interaction: equity return × VXX: captures joint risk‑on / risk‑off behavior between equities and volatility ETN.
    sp_x_vxx = (sp_ret * vxx_ret).rename("SP_RET_X_VXX")

    # 3) Credit spread proxy (high yield vs investment grade)
    credit_spread = (df[params.hyg_symbol] - df[params.lqd_symbol]).rename("CREDIT_SPREAD")
    credit_spread_sq = (credit_spread ** 2).rename("CREDIT_SPREAD_SQ") # Squared credit spread
    credit_spread_change = credit_spread.diff().rename("CREDIT_SPREAD_DIFF")
    
    # Interaction: HY return × credit spread: reflects that HY moves are more important when spreads are already wide.
    hyg_x_spread = (hyg_ret * credit_spread).rename("HYG_RET_X_SPREAD") 

    # 4) Rates and macro
    effr = df["EFFR"]
    baa10y = df["BAA10Y"]

    # 5) Simple transformations / lags (approximate stationarity)
    # Lag macro-type variables by macro_lag_days to simulate info set
    lag = params.macro_lag_days
    effr_lag = effr.shift(lag).rename("EFFR_LAG")
    baa10y_lag = baa10y.shift(lag).rename("BAA10Y_LAG")

    # Assemble into one DataFrame
    features = pd.concat(
        [
            y,
            vol_lag,  
            sp_ret,
            hyg_ret,
            lqd_ret,
            vxx_ret,
            sp_x_vxx,  
            credit_spread,
            credit_spread_sq, 
            credit_spread_change,
            hyg_x_spread,  
            effr_lag,
            baa10y_lag,
        ],
        axis=1,
    )

    # Drop rows where Y is NaN (first vol_window-1 days)
    features = features.dropna(subset=["SPX_21D_REALIZED_VOL"])

    return features

    
def dispersion_diagnostics(df_features: pd.DataFrame) -> None:
    """
    Plot the dispersion series and print the dates and values of the two largest spikes, as a sanity check (e.g., COVID spike and the April 2025 tariff crash).
    """
    y = df_features["SPX_21D_REALIZED_VOL"]

    # plot (unchanged)
    plt.figure(figsize=(10, 4))
    plt.plot(y.index, y.values, label="21D Realized Vol (S&P 500)")
    plt.title("21-Day Realized Volatility of S&P 500")
    plt.ylabel("Annualized Volatility")
    plt.xlabel("Date")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # max over full sample
    max_val = y.max()
    max_date = y.idxmax()

    # second max after some cutoff date, e.g. end of 2020
    cutoff = pd.Timestamp("2021-01-01")
    y_after = y[y.index >= cutoff]
    second_val = y_after.max()
    second_date = y_after.idxmax()

    print(f"Max dispersion (21D realized vol) = {max_val:.4f} on {max_date.date()}")
    print(f"Second max dispersion (21D realized vol) = {second_val:.4f} on {second_date.date()}")


