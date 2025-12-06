# Stressed Market Dispersion Model Development
### Enterprise Risk Framework for Regulatory Stress Testing

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)

---

## Overview

This project implements a novel, fundamentals-driven stress testing framework designed for enterprise financial risk management and regulatory capital adequacy assessment. Developed originally for a global investment bank, this public reconstruction demonstrates an innovative approach to modeling equity dispersion under severe market stress scenarios using elastic net regression with time-series-aware validation.

**Key Innovation**: Unlike traditional autoregressive models with opaque variable selection, this framework uses economically justified market and macroeconomic factors to explain the dispersion of market data, enabling transparent identification of primary risk drivers under different stress regimes (e.g., financial crisis, European debt crisis, COVID-19, inflation shocks).

---


## Business Context & Problem Statement

**Regulatory Requirement**: Financial institutions must demonstrate capital adequacy under stressed market scenarios for prudential valuation and regulatory compliance.

**Technical Challenge**: Existing internal models used ad-hoc autoregressive approaches with:
1. No economic justification for variable selection for independent varaibles
2. convenient data obtained from multiple platforms given the data availability 1 year ago
3. arbitrary selection of dependent variables given the most frequently used through experience without statistic evidence backing
4. Manual, excel-based, error-prone quarterly recalibration processes(2+ hours)
5. No proper time-series validation or train/test splitting
6. Limited interpretability for risk stakeholders

**Solution**: A transparent, production-ready model development and automation framework that:
1. Uses risk-aligned, fundamentals-based feature selection for independent variables
2. avoid data convenience by communicating across functions to ensure data availabilty for the data that we actually need, ensures the data availability inside the firm system, make sure the names of the independent variables to be consistent across teams. centrilizing data source
3. selected dependent variables (dispersions) for each asset class according to risk exposure statistics, evaluated based on the occurance and magnitude of valuation inputs that have material impact of prudent valuation, and further examinate if the dispersion selected behaviors align with expectation that it should spike in stress scenarios like covid time, if not, reselct again iteratively. 
4. Automates quarterly recalibration (reduced processing time by **~95%: from 2 hours to 5 minutes**) solved 57 excel errors through automation (add more enhancements, in recommendation letter from charie)
5. Implements proper time-series cross-validation with elastic net regularization  
6. Provides clear economic interpretation of stress drivers by regime

---

## Methodology

The model development follows a systematic, risk-aligned approach designed from first principles:

![Model Development Workflow](figures/methodology_flowchart.png)

*Figure 1: Complete model development workflow from variable selection to automated deployment*

### Data & Feature Engineering

- **Dependent Variable**: Equity cross-sectional dispersion (21-day rolling realized volatility of S&P 500)
  - Selected based on risk exposure statistics and stress behavior validation
  - Verified alignment with historical stress event of COVID March 2020

- **Independent Variables**: Market and macroeconomic factors driving equity risk/dispersion.
  - Curated from risk team's comprehensive variable list (should I say this?)
  - Examples: VXX (volatility), S&P 500 returns, credit spreads (HYG, LQD), Fed funds rate, BAA-10Y spread (update the complete list)
  - **Daily frequency** chosen to maximize sample size and capture lag dynamics (vs. monthly data).

### Model Development

**Baseline**: Ordinary Least Squares (OLS) regression

**Primary Model**: Elastic Net with TimeSeriesSplit cross-validation
- **Why Elastic Net?**
  - Handles multicollinearity among correlated market factors
  - Embedded feature selection via L1 regularization
  - Improved stability and interpretability in stress regimes
  - Reduces overfitting risk

**Hyperparameter Tuning**:
- Grid search over α (regularization strength) and L1-ratio
- 10-fold time-series cross-validation preserving chronological order
- Prevents data leakage from future to past

**Validation Strategy**:
- Train/test split: 80/20 by time
- Out-of-sample evaluation on recent data (2025 data)
- Separate RMSE metrics for **normal vs. stress regimes** (top 10% volatility periods)

---

## Key Enhancements Over Legacy Model

### 1. Fundamentals-Based Variable Selection
- Transparent alignment with risk team's comprehensive market variable taxonomy
- Ensures data availability within firm systems
- Consistent naming across teams → eliminated mapping errors
- Variables validated by stress behavior (confirmed spikes during COVID)

### 2. Automated Data Ingestion *(Production-Ready)*
- Python ETL pipeline for automated data pulls from internal systems
- Code complete; pending infrastructure configuration
- Public repo demonstrates concept using `yfinance` API for market data (change this)

### 3. Proper Time-Series Validation
- Implemented `TimeSeriesSplit` for chronologically-aware cross-validation
- Train/test split maintains temporal ordering
- Grid search over hyperparameters with automated selection
- Baseline OLS comparison to validate elastic net performance

### 4. Quarterly Recalibration Automation
- **Impact**: Reduced processing time from **2 hours manual → 5 minutes automated**
- Eliminated manual error in variable selection and formula updates
- Enabled more frequent model refresh for improved risk governance
- Modular Python codebase allows seamless integration into production pipelines

---

## Repository Structure

Stressed-Model-development/
│
├── src/ # Modular Python codebase
│ ├── config.py # Model parameters & configuration
│ ├── data_loader.py # Data ingestion (yfinance, FRED API) -- change
│ ├── features.py # Feature engineering pipeline
│ └── model.py # Train/test split, model fitting, evaluation
│
├── notebooks/ # Analysis & visualization notebooks
│ └── Stressed Model Development.ipynb
│
├── figures/ # Generated model diagnostics
│ ├── actual_vs_pred_vol_test.png
│ ├── dispersion_sp500.png
│ └── elastic_net_top_coefs.png
│
├── README.md # This file
├── requirements.txt # Python dependencies
└── LICENSE # MIT License --------------- why????


---

## Installation & Usage

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

Clone repository
git clone https://github.com/YCJG/Stressed-Model-development.git
cd Stressed-Model-development

Install dependencies
pip install -r requirements.txt

Run Jupyter notebook
jupyter notebook notebooks/"Stressed Model Development.ipynb"


### Data Sources (Public Reconstruction)
This public repository uses **proxy data** from publicly available sources to demonstrate methodology while respecting confidentiality:
- **Market data**: `yfinance` (S&P 500, VXX, HYG, LQD) -- change
- **Macro data**: FRED API (Fed funds rate, credit spreads)

**Note**: Internal production model uses proprietary firm data with consistent variable definitions across risk teams.

---

## Design Decisions & Economic Rationale

### Why Daily Data?
Monthly data would severely reduce sample size (~36 observations for 3-year lookback), making lags/rolling windows uninformative. Daily data provides sufficient granularity while smoothing noise via 21-day volatility windows.

### Why 21-Day Rolling Volatility?
Standard monthly volatility measure in finance; aligns with regulatory reporting conventions and captures dispersion during stress events.

### Why Elastic Net vs. Simple OLS?
1. **Multicollinearity**: Market factors (VXX, SPX, credit spreads) are highly correlated
2. **Feature Selection**: L1 penalty automatically identifies most important drivers
3. **Stability**: Ridge component (L2) improves coefficient stability across regimes
4. **Interpretability**: Non-zero coefficients have clear economic meaning for stress scenarios


### Stress Period Identification
- **COVID (March 2020)**: Primary reference for maximum volatility spike
- **April 2025 tariff shock**: the sharp selloff triggered by the new U.S. tariff program and the brief “tariff crash” in global equities.Equity indices including the S&P 500 fell by more than 10% over the following couple of trading days, producing the worst two‑day decline since the early stages of the COVID‑19 crisis, while the VIX and realized volatility jumped to levels not seen since 2020

---

## Impact & Applications

### Operational Impact
- **95% reduction** in quarterly recalibration time (2h → 5min)
- **Eliminated manual errors** in variable selection and formula updates
- **Improved model governance**: Transparent, auditable methodology

### Business Impact
- **Enhanced regulatory compliance**: Transparent stress testing framework for capital adequacy
- **Better risk decision-making**: Clear identification of stress drivers by scenario
- **Scalability**: Framework applicable to other asset classes (FX, rates, credit)

### Use Cases
1. **Regulatory Stress Testing**: CCAR, ICAAP, Dodd-Frank compliance
2. **Prudent Valuation**: Model risk management for fair value adjustments
3. **Capital Planning**: Scenario-based capital buffer determination
4. **Risk Reporting**: Executive dashboards for stress exposure monitoring

---

## Technical Skills Demonstrated

- **Machine Learning**: Scikit-learn (ElasticNetCV, TimeSeriesSplit), regularization techniques
- **Time-Series Analysis**: Rolling windows, lag features, regime identification
- **Financial Modeling**: Volatility estimation, stress scenario design, regulatory frameworks
- **Software Engineering**: Modular Python architecture, reproducible research, version control
- **Data Engineering**: API integration (yfinance, FRED), data validation, feature pipelines -- change yfinance
- **Visualization**: Matplotlib, diagnostic plots, model interpretation
- **Domain Expertise**: Capital markets, risk management, quantitative finance

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -- change

---

## Acknowledgments

This public repository is a **confidential-data-free reconstruction** of work originally developed for enterprise risk management at a global investment bank. All data used here is publicly available; no proprietary information, internal identifiers, or confidential model parameters are disclosed.

**Disclaimer**: This is a personal project for educational and portfolio purposes. It does not represent the views or methodologies of any current or former employer.

---

**Keywords**: Stress Testing, Risk Management, Elastic Net, Time Series, Regulatory Compliance, Capital Adequacy, Machine Learning, Financial Modeling, Python, Scikit-learn







build one tidy DataFrame with:

Y = equity dispersion (e.g., 21-day rolling vol of S&P 500 returns),

X = your macro/market feature set.

Then:

standardize X,

run OLS,

run ElasticNetCV (time-series aware split),

compare RMSE.

Document clearly:

your train/test split logic,

why you chose Elastic Net (handles multicollinearity, feature selection),

how much RMSE improves over OLS.



why each variable makes economic sense as a driver of equity dispersion,

why you use rolling windows and lags (information set at time t),

why you choose specific hyperparameters or let ElasticNetCV tune them,

how you validate (out-of-sample RMSE, maybe a simple time-based split).

Why you chose a particular evaluation horizon (e.g. “to test model robustness during the 2024–2025 high‑vol regime”).

That you considered regime shifts and didn’t just blindly apply a random split—which shows strong quantitative judgment.

