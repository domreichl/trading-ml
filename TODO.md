## TODO
1. tune_models.py
    - "experimentation" with "exp_arima"
    - use optuna together with dvclive
    - parameters:
        - look_back_window_size
        - seasonal_period (sp)
    - goals:
        1. build a solid, reusable experimentation protocol
        2. once I've found the best parameters for AutoArima --> run get precise AutoArima outputs (over all securities & n validations) --> fixate parameters to not use AutoArima, but ARIMA for val_ & prod_
2. integrate 'Models' as described below
3. rerun validation with full date range
4. rerun main pipeline with extended date range

### Models
- LinearNN:
    - as simple baseline (like DLinear, but without decomposition not needed for LogReturns)
    - input: LookBackWindow
    - hidden: DenseLayer with linear activation
    - output: ForecastWindow
- globals models to utilize cross-series information
- TCN: temporal convolutional network -> keras.layers.Conv1d(padding='causal', 'dilation_rate'>1)
- FFNN: N-BEATS
- RNNs: DeepAR, adRNNCell, DA-RNN, MQRNN
- DecisionTree, RandomForest, LightGBX & XGBoost as MultiClassClassifiers for SignPrediction (multi-class for full ForecastWindow)
- sktime docs: https://www.sktime.net/en/stable/api_reference/forecasting.html
- models specifically for leveraged products:
    - knock-outs:
        - 1-4 day forecast horizon
        - analysis/prediction of volatility/likelihood of knock-out
        - high penalty for potential knock-out
    - factors:
        - evaluation with entire forecast window (because the full path is relevant)
        - probably focus on multi-class sign prediction
        - optimize model for non-linear paths (the model shouldn't systematically just all up/down!)
        - recommendation: invest when every prediction in path has the same sign with high confidence

### Model Tuning
- tuning of all models (except transformer)
    - especially ETS, LightGBM, and XGBoost
    - RNN (Zellen unwesentlich: einfach 128)
- Pinball loss
- COCOB instead of Adam optimizer
- stacking (of RNNs)

### Data Sources
- https://www.wienerborse.at/aktien-prime-market/
    - 43 prime, 22 standard, 7 direct+, 21 direct, 780 global
    - use all time series with T > LBWS
- stick to eurozone ISINs to minimize currency risk (ATX < ATX+DAX < MSCI EMU < EMU part of Stoxx Europe 600)

### Data Preprocessing
- optimal number of test days
- optimal look back window size
- normalization methods
- with/without market index

### Recommendation
- improve ensembling algorithm
- add technical indicators