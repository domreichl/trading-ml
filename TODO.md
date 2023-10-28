## TODO
1. hypertuning:
  - LGBMRegressor
    - Base: LBWS only
    - MLForecast:
      - lags
      - lag_transforms
    - LGBMRegressor:
      - boosting_type
      - num_leaves
      - learning_rate
      - n_estimators
  - XGBRegressor:
    - Base: LBWS only
    - MLForecast:
      - lags
      - lag_transforms
    - XGBRegressor:
      - booster
      - "n_estimators": [100, 200, 300, 400],
      - "learning_rate": [0.001, 0.005, 0.01, 0.05],
      - "max_depth": [8, 10, 12, 15],
      - "gamma": [0.001, 0.005, 0.01, 0.02],
  - update 'look_back_window_size' in models.yaml
2. rerun validation with full date range
3. rerun main pipeline with extended date range

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
    - neural_nets.py: LocalLinearClassifier, GlobalLinearClassifier
2. deep neural nets:
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