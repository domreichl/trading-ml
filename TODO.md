## TODO
1. DVC tracking:
    - goal: execute the pipeline for every week until End Date == 13.10.
    - think about how to best track & compare performances with DVC:
        - [!!] use 'dvclive' (see bookmarks folder "DVCLive")
        - use DVC 'params': "start_date" & "end_date"
        - track metrics in .json-format with DVCLive (keeping the CSVs for now)
        - use DVC 'plots'
        - use DVC 'artifacts' for model metadata
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
    - trading steps:
        - analyze & visualize all results files
        - briefly research / think about fundamentals (news about the stock)
        - open/close a short/long position
        - track trade in database
3. serious modeling:
    - new folder "src/optimization" or "src/experimentation" with dvc.yaml
        - experimentation pipeline ('dvc exp [run, show, diff]' commands) with "dev_" or "exp_" models
        - update, integrate, or delete optimization.py
        - uninstall optuna
    - improve all model prototypes
    - train with or without market index?
    - revisit my deep forecasting presentation to implement additional models
    - potential libraries: statsforecast, sktime
    - additional/alternative data sources and preprocessing
    - statistical testing for model comparision: test significance of differences in performance (e.g., RMSE) between models for n validations