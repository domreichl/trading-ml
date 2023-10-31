## TODO
1. extend data:
    - https://www.wienerborse.at/aktien-prime-market/
    - 43 prime, 22 standard, 7 direct plus, 21 direct
    - use all with T > 260
2. trading + performance analysis

### Models
1. multi class classifiers for direct sign prediction:
    - sklearn.linear_model.LogisticRegression
    - sklearn.ensemble.RandomForestClassifier
    - LGBMClassifier
    - XGBClassifier
    - neural_nets.py: LocalLinearClassifier, GlobalLinearClassifier
2. deep neural nets:
    - add indicators (at least 3 EMAs for each TS) as additional input features
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