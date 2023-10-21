## TODO
0. current:
    - analyze the results with DVC VSCode extension + Iterative Studio
    - execute "main" for every week until End Date == 20.10.
    - for every week: simulate a trade & track trading performance -> update all relevant performance files
1. Simulate trades for:
    - 230922 (see Desktop)
    - 230929 (current git tag)
2. serious modeling:
    - pipeline "experimentation" with "exp_" data and models
        - see bookmarks folder "DVCLive" & "ModelRegistry"
        - commands: 'dvc exp [run, show, diff]'
    - improve all model prototypes
    - models for leveraged positions:
        - 1-4 day forecast horizon (open on Mon/Tue, sell on Fri)
        - Faktorzertifikat:
            - prediction of %return for each day --> optimize path, e.g., every day positive
            - could also be multi-day sign prediction
            - warning: make sure model are capable of predicting non-linear paths! (just just all up/down)
        - Knock-out-Zertifikat:
            - prediction of both price & voliatility
            - during validation, minimize error w.r.t. price declines (or increases for short positions)
            - during price+volatility forecast analysis, include a high penalty for potential knock-out
    - revisit my deep forecasting presentation to implement additional models
    - potential libraries: statsforecast, sktime
    - additional data sources
        - https://www.wienerborse.at/aktien-prime-market/
            - 43 prime, 22 standard, 7 direct+, 21 direct, 780 global
            - use all time series with T > LBWS
        - stick to eurozone ISINs to minimize currency risk (ATX < ATX+DAX < MSCI EMU < EMU part of Stoxx Europe 600)
    - alternative preprocessing
        - train with or without market index?
    - statistical testing for model comparision: test significance of differences in performance (e.g., RMSE) between models for n validations