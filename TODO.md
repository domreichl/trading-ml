## TODO
0. read DVC docs & implement
    - https://dvc.org/doc/start/data-management/metrics-parameters-plots
    - https://dvc.org/doc/start/experiments
1. execute the pipeline for every week until End Date == 13.10.
    - every week:
        - simulate a trade
        - track performance in database
        - update all relevant performance files
        - analyze the results
    - analyze overall results
    - update the frontend
2. (22.10.) research taxes & brokers >> extend date range >> execute pipeline >> first trade
    - trading steps:
        - analyze & visualize all results files
        - briefly research / think about fundamentals (news about the stock)
        - open/close a short/long position
        - track trade in database
3. serious modeling:
    - update & refactor optimization.py
    - improve all model prototypes
    - train with or without market index?
    - revisit my deep forecasting presentation to implement additional models
    - potential libraries: statsforecast, sktime
    - additional/alternative data sources and preprocessing
    - statistical testing for model comparision: test significance of differences in performance (e.g., RMSE) between models for n validations