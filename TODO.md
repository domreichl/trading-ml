## TODO
1. Prefect for workflow automation
    - pipeline: Data Update >> Preprocessing >> Model Evaluation >> Model Uptraining >> Model Evaluation >> Model Selection >> Model Prediction >> Visualization Update >> Trading >> Trading Results >> Trading Visualization Update
    - automated data updates with weekly training & new predictions every Sunday
    - track average metrics over all tests (plot in frontend instead of performance.csv)
    - run Docker Image (saved on dockerhub) serverless with Google Cloud Run
2. (24.10.) update datasets -> retrain all models -> first investment based on recommendation_open.py
3. serious modeling:
    - improve all model prototypes
    - revisit my deep forecasting presentation to implement additional models
    - potential libraries: statsforecast, sktime
    - additional/alternative data sources and preprocessing
    - statistical testing for model comparision: test significance of differences in performance (e.g., RMSE) between models for n validations