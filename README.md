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
1. Update *config.yaml*
2. Download & preprocess data: `trading-ml prepare`
3. Train a model: `trading-ml train [model_name]`
4. Validate a model on the train set: `trading-ml validate [model_name]`
5. Evaluate a model on the test set: `trading-ml test [model_name]`
6. Generate test set predictions: `trading-ml predict [model_name] [ISIN]`
7. Generate an out-of-sample forecast: `trading-ml forecast [model_name] [ISIN]`
8. Get a trading recommendation:
    - `trading-ml recommend-open [position_type] [optimize]`
    - `trading-ml recommend-close [position_type] [ISIN]`
9. Plot performance metrics:
    - `trading-ml plot-metrics optimization`
    - `trading-ml plot-metrics validation`
    - `trading-ml plot-metrics evaluation`
10. Explore trading strategies: `trading-ml backtest`
11. Fetch trades to compute statistics: `trading-ml fetch-trades`
12. Run production workflow: `dvc repro` in src/pipeline

### Parameters
- [model_name] ∈ config.yaml:models
- [ISIN] ∈ config.yaml:securities
- [position_type] ∈ {short, long}
- [optimize] ∈ {risk, return}

### Visualization
- https://dominicreichl.com/ml/stock-price-prediction/

### Warning
- This library is intended for technical purposes only and not to be financial advice.
- Beware of the survivorship bias due to selection of stocks with long-term data from market indices.
- Beware of the model selection bias due to multiple validations, evaluations, and backtests.
