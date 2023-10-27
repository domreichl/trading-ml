## TODO
1. hypertuning:
  - simple_regression_net
  - recurrent_regression_net
  - LGBMRegressor: https://github.com/optuna/optuna-examples/blob/main/lightgbm/lightgbm_integration.py
  - XGBRegressor:
    - https://www.kaggle.com/code/mtszkw/xgboost-for-stock-trend-prices-prediction
    - https://github.com/optuna/optuna-examples/blob/main/xgboost/xgboost_integration.py
2. fixate optimale parameters for val_ & prod_ models:
    - LookBackWindowSize:
      - 520 (2 years):
        - arima (1300 would be just as good, but less efficient)
        - prophet (exactly the best)
      - 780 (3 years):
        - moving_average_recursive
        - exponential_smoothing
3. refactor validation loops
4. rerun validation with full date range
5. rerun main pipeline with extended date range


### Data
- https://www.wienerborse.at/aktien-prime-market/
    - 43 prime, 22 standard, 7 direct+, 21 direct, 780 global
    - use all time series with T > LBWS
- stick to eurozone ISINs to minimize currency risk (ATX < ATX+DAX < MSCI EMU < EMU part of Stoxx Europe 600)

### Models
1. multi class classifiers for direct sign prediction:
    - sklearn.linear_model.LogisticRegression
    - sklearn.ensemble.RandomForestClassifier
    - LGBMClassifier
    - XGBClassifier
    - neural_networks.py: LocalLinearClassifier, GlobalLinearClassifier
2. deep neural networks:
    - one model in PyTorch
    - TCN: temporal convolutional network -> keras.layers.Conv1d(padding='causal', 'dilation_rate'>1)
    - FFNN: N-BEATS
    - RNNs: DeepAR, adRNNCell, DA-RNN, MQRNN
3. sktime: EnsembleForecaster & StackingForecaster
4. models specifically for leveraged products:
    - knock-outs:
        - 1-4 day forecast horizon
        - analysis/prediction of volatility/likelihood of knock-out
        - high penalty for potential knock-out
    - factors:
        - evaluation with entire forecast window (because the full path is relevant)
        - probably focus on multi-class sign prediction
        - optimize model for non-linear paths (the model shouldn't systematically just all up/down!)
        - recommendation: invest when every prediction in path has the same sign with high confidence

### Recommendation
- improve ensembling algorithm
- add technical indicators