## TODO
1. DVC tracking:
    - goal: execute the pipeline for every week until End Date == 13.10.
    - think about how to best track & compare performances with DVC:
        - [!!] use 'dvclive' (see bookmarks folder "DVCLive")
        - use DVC 'params': "start_date" & "end_date"
        - track metrics in .json-format with DVCLive (keeping the CSVs for now)
        - use DVC 'plots'
        - use DVC 'artifacts' for model metadata
    - new folder "src/optimization" or "src/experimentation" with dvc.yaml
        - experimentation pipeline ('dvc exp [run, show, diff]' commands) with "dev_" or "exp_" models
        - update, integrate, or delete optimization.py
    - for every week:
        - simulate a trade
        - track trading performance
            - probably remove database connection and use CSV instead!?
            - refactor trades.py
            - update all relevant performance files
    - analyze the results
        - check out the DVC VSCode extension (Iterative Studio not needed - that's mainly for collaboration)
    - update the frontend
2. (22.10.) research taxes & brokers >> extend date range >> execute pipeline >> first trade
    - EasyBank: best fees for Wiener Börse (4,95€) & steuereinfach
    - trading steps:
        - analyze & visualize all results files
        - briefly research / think about fundamentals (news about the stock)
        - open/close a short/long position
        - track trade in database
3. serious modeling:
    - improve all model prototypes
    - train with or without market index?
    - revisit my deep forecasting presentation to implement additional models
    - potential libraries: statsforecast, sktime
    - additional/alternative data sources and preprocessing
        - https://www.wienerborse.at/aktien-prime-market/
            - 43 prime, 22 standard, 7 direct+, 21 direct, 780 global
            - use all time series with T > LBWS
            - also 6-7k certificates!
    - statistical testing for model comparision: test significance of differences in performance (e.g., RMSE) between models for n validations