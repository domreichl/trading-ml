## TODO
1. rename & refactor files according to the pipeline below:
    - scripts in /src/pipeline:
        - test.py -> results/test.csv & results/predictions.csv
        - select.py -> results/selection.csv
            - script to find the top 3 models
        - forecast.py -> results/forecast.csv
            - to retrain/uptrain the top 3 models anew with the total data
        - recommend.py -> recommendation.csv        
            - new script based on "recommendation_open.py" that recommends a stock
            - should also have the OPTIONAL functionality (defined in config) of "recommendation_close.py"
            - persist detailed results in a csv file
    - refactor all scripts in /src (esp. cli & prediction.py) to keep files all clean
        - goal: keep only cli.py in /src -> run all scripts for experimentation/development via cli
        - all other code in /src should be put either into cli.py or into utils
    - update Readme (esp. for cli)
2. Workflow automation
    - pipeline: extend date range >> prepare.py >> train.py >> validate.py >> test.py >> select.py >> forecast.py >> recommend.py >> trading (see below) >> frontend (see below) >> update dvc & git (automate as well?)
        - trading: (not automated):
            - check the results in test.csv, selection.csv, forecast.csv & recommendation.csv
            - perhaps also visualize the results
            - briefly research / think about fundamentals (news about the stock)
            - open/close a short/long position
            - track the trade in a database
        - frontend: ensure it is properly updated:
            - especially validation.csv, test.csv, predictions.csv, forecast.csv, recommendation.csv, trades.csv, trading_statistics.csv
    - automated data updates with weekly training & new predictions every Sunday
3. (24.10.) update datasets -> retrain all models -> first investment based on recommendation_open.py
4. serious modeling:
    - improve all model prototypes
    - train with or without market index?
    - revisit my deep forecasting presentation to implement additional models
    - potential libraries: statsforecast, sktime
    - additional/alternative data sources and preprocessing
    - statistical testing for model comparision: test significance of differences in performance (e.g., RMSE) between models for n validations