## TODO
0. recommendation_sell.py
1. enrich the weekly prediction-based trading strategy
    - https://learn.bybit.com/indicators/best-swing-trading-indicators/
    - https://www.ig.com/en/trading-strategies/10-trading-indicators-every-trader-should-know-190604
    - potential libraries: TA, Backtest
    - stay invested until a top model predicts a price drop (compute new predictions every 1-5 days)
    - use 2-6 technical indicators for further guidance
2. Prefect for workflow automation
    - pipeline: Data Update >> Preprocessing >> Model Evaluation >> Model Uptraining >> Model Evaluation >> Model Selection >> Model Prediction >> Visualization Update >> Trading >> Trading Results >> Trading Visualization Update
    - automated data updates with weekly training & new predictions every Sunday
    - track average metrics over all tests (plot in frontend instead of performance.csv)
    - run Docker Image (saved on dockerhub) serverless with Google Cloud Run
3. serious modeling:
    - improve all model prototypes
    - revisit my deep forecasting presentation to implement additional models
    - potential libraries: statsforecast, sktime
    - additional/alternative data sources and preprocessing
    - statistical testing for model comparision: test significance of differences in performance (e.g., RMSE) between models for n validations
