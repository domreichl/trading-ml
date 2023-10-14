# trading-ml
## Machine Learning for Stock Trading

### Installation
1. Install virtual environment: `python -m venv .venv`
2. Activate virtual evironment:
    - Linux: `source .venv/bin/activate`
    - Windows: `.venv\Scripts\activate.bat`
3. Install piptools: `pip install pip-tools`
4. Compile project: `python -m piptools compile pyproject.toml`
5. Install project: `python -m pip install -e .`

### Usage
1. Specify parameters for data processing and model training in *config.py*
2. Download & preprocess data: `trading-ml prepare`
3. Explore trading strategies: `trading-ml backtest`
4. Validate a model: `trading-ml validate [model_name]`

5. Plot performance metrics:
    - `trading-ml plot-metrics optimization`
    - `trading-ml plot-metrics validation`
    - `trading-ml plot-metrics evaluation`
6. Generate predictions (test set) or a forecast (out-of-sample):
    - `trading-ml predict [model_name] [ISIN]`
    - `trading-ml forecast [model_name] [ISIN]`

7. Get trading recommendation:
    - `trading-ml recommend-open [position_type] [optimize]`

### Parameters
- [model_name] ∈ {src/config/config.py/model_config/names}
- [ISIN] ∈ {src/config/securities.json}
- [position_type] ∈ {short, long}
- [optimize] ∈ {risk, return}

### Visualization
- https://dominicreichl.com/ml/stock-price-prediction/

### Bias Warning
- Survivorship bias due to selection of stocks with long-term data from market indices
- Model selection bias due to multiple evaluations and backtests
