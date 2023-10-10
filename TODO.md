## TODO
1. Prefect for workflow automation
    - pipeline: Data Update >> Preprocessing >> Model Evaluation >> Model Uptraining >> Model Evaluation >> Model Selection >> Model Prediction >> Visualization Update >> Trading >> Trading Results >> Trading Visualization Update
    - automated data updates with weekly training & new predictions every Sunday
    - track average metrics over all tests (plot in frontend instead of performance.csv)
    - execute serverless in cloud (e.g. AWS Lambda) or on my krystal server with a Flask app?
        - https://prefecthq.github.io/prefect-aws/
2. serious modeling:
    - improve all model prototypes
    - revisit my deep forecasting presentation to implement additional models
    - potential libraries: statsforecast, sktime
    - additional/alternative data sources and preprocessing
    - statistical testing for model comparision: test significance of differences in performance (e.g., RMSE) between models for n validations
3. develop and backtest real prediction-based trading strategies
    - research weekly/biweekly swing trading strategies and indicators
    - potential libraries: TA, Backtest
