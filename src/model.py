import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from .config import ModelParams
from typing import Tuple
import matplotlib.pyplot as plt

def time_series_train_test_split(
    df_features: pd.DataFrame, params: ModelParams
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    """
    Split features into train/test sets by time, preserving chronological order.
    """
    df_features = df_features.dropna()
    n = len(df_features)
    test_size = int(np.floor(params.test_size_fraction * n))
    train_size = n - test_size

    train = df_features.iloc[:train_size]
    test = df_features.iloc[train_size:]

    X_train = train.drop(columns=["SPX_21D_REALIZED_VOL"])
    y_train = train["SPX_21D_REALIZED_VOL"]
    X_test = test.drop(columns=["SPX_21D_REALIZED_VOL"])
    y_test = test["SPX_21D_REALIZED_VOL"]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train period: {X_train.index[0].date()} to {X_train.index[-1].date()}")
    print(f"Test period:  {X_test.index[0].date()} to {X_test.index[-1].date()}")
    return X_train, X_test, y_train, y_test


def fit_models_with_timeseries_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: ModelParams,
):
    """
    Fit:
    - Baseline Linear Regression (OLS)
    - Elastic Net with TimeSeriesSplit CV (over l1_ratio and alpha grid)

    Returns:
    - dict with models, RMSEs, best hyperparams, coefficients, and CV summary.
    """

    # ----------------------------
    # Standardize features (fit only on train)
    # ----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----------------------------
    # Baseline OLS
    # ----------------------------
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train)
    y_pred_ols = ols.predict(X_test_scaled)
    rmse_ols = math.sqrt(mean_squared_error(y_test, y_pred_ols))

    print(f"\nBaseline OLS Test RMSE: {rmse_ols:.6f}")

    # ----------------------------
    # Elastic Net with TimeSeriesSplit
    # ----------------------------
    tscv = TimeSeriesSplit(n_splits=params.n_splits)

    enet = ElasticNetCV(
        l1_ratio=params.l1_ratios,
        alphas=params.alphas,
        cv=tscv,
        max_iter=params.max_iter,
        n_jobs=-1,
    )

    enet.fit(X_train_scaled, y_train)
    y_pred_enet = enet.predict(X_test_scaled)
    rmse_enet = math.sqrt(mean_squared_error(y_test, y_pred_enet))

    print(f"Elastic Net Test RMSE: {rmse_enet:.6f}")
    print(f"Best l1_ratio: {enet.l1_ratio_}")
    print(f"Best alpha:    {enet.alpha_}")

    # ----------------------------
    # Collect coefficient info
    # ----------------------------
    coef_series = pd.Series(enet.coef_, index=X_train.columns, name="coefficient")
    # Drop coefficients that are (almost) zero
    coef_nonzero = coef_series[coef_series.abs() > 1e-6]

    # Sort remaining features by absolute size
    coef_nonzero = coef_nonzero.sort_values(key=lambda s: s.abs(), ascending=False)

    # Top-N non-zero features
    top_features = coef_nonzero.head(params.top_n_features)

    # ----------------------------
    # Build a small "hyperparameter + RMSE" summary table from cv_results
    # Note: ElasticNetCV does not expose full grid results as cleanly as GridSearchCV.
    # We'll approximate by evaluating the mean MSE path.
    # ----------------------------
    # mse_path_: shape = (n_alphas, n_folds), for each l1_ratio separately.
    # We'll flatten (l1_ratio, alpha) pairs with their mean CV error.
    rows = []
    for i, l1 in enumerate(np.atleast_1d(enet.l1_ratio)):
        # If a single l1_ratio is used, mse_path_ is 2D; if multiple, it's 3D.
        mse_path = enet.mse_path_
        if mse_path.ndim == 3:
            # shape: (n_l1_ratio, n_alpha, n_folds)
            mse_l1 = mse_path[i]
        else:
            # only one l1_ratio, broadcast
            mse_l1 = mse_path

        mean_mse = mse_l1.mean(axis=1)  # average over folds
        for alpha_val, mse_val in zip(enet.alphas_, mean_mse):
            rows.append(
                {
                    "l1_ratio": float(l1),
                    "alpha": float(alpha_val),
                    "cv_mse": float(mse_val),
                    "cv_rmse": float(math.sqrt(mse_val)),
                }
            )

    cv_df = pd.DataFrame(rows)
    cv_df = cv_df.sort_values("cv_rmse", ascending=True).reset_index(drop=True)
    cv_top = cv_df.head(params.top_n_models)

    results = {
        "scaler": scaler,
        "ols_model": ols,
        "enet_model": enet,
        "rmse_ols": rmse_ols,
        "rmse_enet": rmse_enet,
        "coef_series": coef_series,
        "top_features": top_features,
        "cv_results": cv_df,
        "cv_top": cv_top,
        "y_test": y_test,
        "y_pred_ols": y_pred_ols,
        "y_pred_enet": y_pred_enet,
    }

    return results

def report_results(results, params: ModelParams):
    """
    Print and plot key outputs:
    - RMSE comparison
    - Top features
    - Top CV models
    - Normal vs stress regime RMSE
    - Predicted vs actual chart
    """
    print("\n=== RMSE COMPARISON (TEST SET) ===")
    print(f"OLS RMSE:        {results['rmse_ols']:.6f}")
    print(f"Elastic Net RMSE:{results['rmse_enet']:.6f}")

    print("\n=== TOP FEATURES BY |COEFFICIENT| (Elastic Net) ===")
    print(results["top_features"])

    print("\n=== TOP CV MODELS (Elastic Net) ===")
    print(results["cv_top"])

    # Extract series for convenience
    y_test = results["y_test"]
    y_pred_enet = results["y_pred_enet"]
    y_pred_ols = results["y_pred_ols"]

    
    # Separate “normal” and “stress” evaluation. Instead of one RMSE for the whole test set, compute:
    # RMSE on normal days (vol below some threshold).
    # RMSE on stress days (top X% of vol or around a known crisis window).
    
    # -------------------------------------------------
    # Normal vs stress regime evaluation
    # -------------------------------------------------
    # Define stress as top 10% of test-set volatility values
    stress_threshold = y_test.quantile(0.9)
    stress_mask = y_test > stress_threshold

    rmse_enet_normal = math.sqrt(
        mean_squared_error(y_test[~stress_mask], y_pred_enet[~stress_mask])
    )
    rmse_enet_stress = math.sqrt(
        mean_squared_error(y_test[stress_mask], y_pred_enet[stress_mask])
    )

    rmse_ols_normal = math.sqrt(
        mean_squared_error(y_test[~stress_mask], y_pred_ols[~stress_mask])
    )
    rmse_ols_stress = math.sqrt(
        mean_squared_error(y_test[stress_mask], y_pred_ols[stress_mask])
    )

    print("\n=== NORMAL vs STRESS RMSE (TEST SET) ===")
    print(f"Normal regime (<= {stress_threshold:.4f}):")
    print(f"  OLS RMSE:        {rmse_ols_normal:.6f}")
    print(f"  Elastic Net RMSE:{rmse_enet_normal:.6f}")
    print(f"Stress regime (> {stress_threshold:.4f}):")
    print(f"  OLS RMSE:        {rmse_ols_stress:.6f}")
    print(f"  Elastic Net RMSE:{rmse_enet_stress:.6f}")

    # -------------------------------------------------
    # Predicted vs actual plot
    # -------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(y_test.index, y_test.values, label="Actual Vol", color="black", linewidth=1.5)
    plt.plot(y_test.index, y_pred_ols, label="OLS Predicted", alpha=0.7)
    plt.plot(y_test.index, y_pred_enet, label="Elastic Net Predicted", alpha=0.7)
    plt.title("Actual vs Predicted 21D Realized Vol (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Volatility (same scale as target)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Regression equation (top features only, ordered by importance)
    print("\n=== ELASTIC NET REGRESSION (TOP FEATURES ONLY) ===")
    eq_terms = []
    for feature, coef in results["top_features"].items():
        eq_terms.append(f"{coef:.4f} * {feature}")
    eq_str = " + ".join(eq_terms)
    print("SPX_21D_REALIZED_VOL ≈ " + eq_str)
