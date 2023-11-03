import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["figure.figsize"] = 20, 10

from utils.file_handling import ResultsHandler


def plot_overfitting() -> None:
    val = ResultsHandler().load_csv_results("validation_results")
    val["Model"] = val["Model"].str.replace("val_", "")
    val["Data"] = "train"
    test = ResultsHandler().load_csv_results("test_metrics")
    test["Model"] = test["Model"].str.replace("main_", "")
    test["Data"] = "test"
    fig, axs = plt.subplots(2, 1)
    sns.barplot(
        ax=axs[0],
        x="Model",
        y="RMSE",
        hue="Data",
        data=pd.concat(
            [
                val[["Model", "Data", "RMSE"]],
                test[test["Metric"] == "RMSE"][
                    ["Model", "Data", "Metric", "Score"]
                ].rename(columns={"Score": "RMSE"}),
            ]
        ),
    )
    sns.barplot(
        ax=axs[1],
        x="Model",
        y="PredictiveScore",
        hue="Data",
        data=pd.concat(
            [
                val[["Model", "Data", "PredictiveScore"]],
                test[test["Metric"] == "PredictiveScore"][
                    ["Model", "Data", "Metric", "Score"]
                ].rename(columns={"Score": "PredictiveScore"}),
            ]
        ),
    )
    plt.suptitle(f"Overfitting between Train Set Validation and Test Set Evaluation")
    plt.show()


def plot_validation_metrics() -> None:
    validation = ResultsHandler().load_json_results("validation_metrics")
    models, metrics, scores = [], [], []
    for model, subdict in validation.items():
        for metric, score in subdict.items():
            models.append(model)
            metrics.append(metric)
            scores.append(score)
    df = pd.DataFrame({"Model": models, "Metric": metrics, "Score": scores})
    fig, axs = plt.subplots(df["Metric"].nunique(), 1)
    for i, metric in enumerate(df["Metric"].unique()):
        sns.barplot(
            ax=axs[i],
            x="Model",
            y="Score",
            hue="Metric",
            data=df[df["Metric"] == metric],
            palette="Blues" if metric == "F1" else "Oranges",
        )
    plt.suptitle(f"Train Set Validation Error for {df['Model'].nunique()} Models")
    plt.show()


def plot_test_metrics() -> None:
    df = ResultsHandler().load_csv_results("test_metrics")
    fig, axs = plt.subplots(2, 1)
    barplot_cfg = {"x": "Metric", "y": "Score", "hue": "Model"}
    sns.barplot(
        ax=axs[0],
        data=df[df["Target"] == "Sign"],
        palette="Blues",
        **barplot_cfg,
    ).set(xlabel=None)
    sns.barplot(
        ax=axs[1],
        data=df[(df["Target"] != "Sign") & (~df["Metric"].isin(["MAE", "RMSE"]))],
        palette="Oranges",
        **barplot_cfg,
    )
    plt.suptitle(
        f"Test Set Performance of {df['Model'].nunique()} Prediction Models on Various Metrics",
        y=0.9,
    )
    for i, ax in enumerate(axs):
        if i in [1, 2]:
            decimals = 3
        else:
            decimals = 2
        for patch in ax.patches:
            ax.text(
                patch.get_x() + patch.get_width() / (2 + (len(ax.patches) - 1) * 0.1),
                patch.get_height() / 2.0,
                round(patch.get_height(), decimals),
            )
    plt.show()


def plot_price_predictions(df: pd.DataFrame) -> None:
    if len(df) == 0:
        raise Exception(f"No predictions were generated for ISIN {ts_name}")
    ts_name = df["ISIN"].iloc[0]
    actual_prices = pd.DataFrame(
        {
            "PricePredicted": list(df["Price"][: df["Date"].nunique()]),
            "Date": list(df["Date"][: df["Date"].nunique()]),
        }
    )
    actual_prices["ISIN"] = ts_name
    actual_prices["Model"] = "ACTUAL PRICE"
    df = pd.concat([actual_prices, df])
    df = df[["ISIN", "Date", "PricePredicted", "Model"]]
    linepplot = sns.lineplot(
        x="Date", y="PricePredicted", hue="Model", style="Model", size="Model", data=df
    )
    linepplot.set(ylabel="Price [€]")
    plt.title(
        f"Stock Price Prediction for {ts_name} over the next {df['Date'].nunique()} Business Days"
    )
    plt.show()


def plot_price_forecast(df: pd.DataFrame) -> None:
    if len(df) == 0:
        raise Exception(f"No forecast was generated for ISIN {ts_name}")
    ts_name = df["ISIN"].iloc[0]
    linepplot = sns.lineplot(
        x="Date", y="Price", hue="Model", style="Model", size="Model", data=df
    )
    linepplot.set(ylabel="Price [€]")
    plt.title(
        f"Stock Price Forecast for {ts_name} over the next {df['Date'].nunique()} Business Days"
    )
    plt.show()
