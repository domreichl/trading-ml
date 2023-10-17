import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["figure.figsize"] = 20, 10

from utils.file_handling import load_csv_results


def plot_optimization_metrics() -> None:
    df = load_csv_results("optimization")
    barplot = sns.barplot(
        x="look_back_window_size", y="RMSE", hue="Model", data=df, palette="Oranges"
    )
    barplot.set(ylabel="RMSE (log scale)", yscale="log")
    plt.title(
        f"Validation Error for Hyperparameter Optimization with {df['Model'].nunique()} Prediction Models"
    )
    plt.show()


def plot_validation_metrics() -> None:
    df = load_csv_results("validation")
    mae = df[["Model", "MAE"]].rename(columns={"MAE": "Score"})
    mae["Metric"] = "MAE"
    rmse = df[["Model", "RMSE"]].rename(columns={"RMSE": "Score"})
    rmse["Metric"] = "RMSE"
    df = pd.concat([mae, rmse])
    df["Score"] = df["Score"]
    barplot = sns.barplot(
        x="Model", y="Score", hue="Metric", data=df, palette="Oranges"
    )
    barplot.set(ylabel="Score (log scale)", yscale="log")
    plt.title(
        f"Train Set Validation Error for {df['Model'].nunique()} Optimized Prediction Models"
    )
    plt.show()


def plot_prediction_metrics() -> None:
    df = load_csv_results("performance")
    fig, axs = plt.subplots(4, 1)
    barplot_cfg = {"x": "Model", "y": "Score", "hue": "Metric"}
    sns.barplot(
        ax=axs[0],
        data=df[df["Target"] == "Sign"],
        palette="Blues",
        **barplot_cfg,
    ).set(xlabel=None)
    sns.barplot(
        ax=axs[1],
        data=df[(df["Target"] == "Return") & (~df["Metric"].isin(["MASE", "RMSSE"]))],
        palette="Oranges",
        **barplot_cfg,
    ).set(xlabel=None)
    sns.barplot(
        ax=axs[2],
        data=df[(df["Target"] == "Return") & (df["Metric"].isin(["MASE", "RMSSE"]))],
        palette="Oranges",
        **barplot_cfg,
    ).set(xlabel=None)
    sns.barplot(
        ax=axs[3],
        data=df[df["Target"] == "Price"],
        palette="Oranges",
        **barplot_cfg,
    ).set(ylabel="Score [%]")
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


if __name__ == "__main__":
    plot_optimization_metrics()
    plot_validation_metrics()
    plot_prediction_metrics()
    predictions = load_csv_results("predictions")
    forecast = load_csv_results("forecast")
    ts_name = random.choice(list(predictions["ISIN"].unique()))
    plot_price_predictions(predictions[predictions["ISIN"] == ts_name])
    plot_price_forecast(forecast[forecast["ISIN"] == ts_name])
