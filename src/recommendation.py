import os
import pandas as pd

from config.config import paths


def analyze_predictions():
    df = pd.read_csv(os.path.join(paths["results"], "performance.csv"))
    pass


"""
    rank ISINs by
        (a) number of models predictions profits
            with models weighted by metrics
                /RMSE
                *F1
            calculated from validation
            & previous week's performance
            & (later) trading performance
        (b) % of price increase expected
    for a single point metric, compute (a)*(b)
        but also analyze all factors separately
    print separate factors, recommendation formula, and final result
    invest in ISN with highest final result (rank/custom metric)
    stay invested until some models predict a price drop (compute new predictions every 1-5 days)
    use 2-6 technical indicators for further guidance
"""
