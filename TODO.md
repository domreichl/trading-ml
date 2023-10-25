## TODO
0. try ARIMA with close prices instead of log returns
1. fixate ARIMA parameters for val_ & prod_ --> max lbws & (0,0,0)x(0,0,0)
2. optimize the other 5 tunable models as well (moving_averge_recursive is done)
3. integrate 'Models' as described below
4. rerun validation with full date range
5. rerun main pipeline with extended date range

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
- one model in PyTorch
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