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

### DVC Pipelines
1. Main pipeline in src/main
2. Validation pipeline in src/validation
3. Experimentation pipeline src/experimentation

### CLI Usage
1. Download and preprocess cli_ data: `trading-ml prepare`
2. Train a cli_ model: `trading-ml train [model_name]`
3. Validate a cli_ model on the train set: `trading-ml validate [model_name]`
4. Evaluate a cli_ model on the test set: `trading-ml test [model_name]`
5. Generate cli_ test set predictions: `trading-ml predict [model_name] [ISIN]`
6. Use prod_ models to generate an out-of-sample forecast: `trading-ml forecast [ISIN]`
7. Get a trading recommendation from prod_ models:
    - `trading-ml recommend-open [position_type] [optimize]`
    - `trading-ml recommend-close [position_type] [ISIN]`
8. Plot performance metrics: `trading-ml plot-metrics [metrics_type]`
9. Explore trading strategies: `trading-ml backtest`
10. Fetch trades to compute statistics: `trading-ml fetch-trades`

### CLI Parameters
- [model_name] ∈ model_config.yaml:models
- [ISIN] ∈ data_config.yaml:securities
- [position_type] ∈ {short, long}
- [optimize] ∈ {risk, return}
- [metrics_type] ∈ {validation, evaluation}

### Visualization
- https://dominicreichl.com/ml/stock-price-prediction/

### Warning
- This library is intended for technical purposes only and not to be financial advice.
- Beware of the survivorship bias due to selection of stocks with long-term data from market indices.
- Beware of the model selection bias due to multiple validations, evaluations, and backtests.
