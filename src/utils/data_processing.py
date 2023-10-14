import numpy as np
import pandas as pd


def stack_array_from_dict(dictionary: dict, axis: int) -> np.array:
    return np.stack(list(dictionary.values()), axis)


def get_signs_from_prices(prices: dict) -> np.array:
    return np.concatenate(
        [np.array(np.array(v)[1:] > np.array(v)[:-1], int) for v in prices.values()],
        0,
    )


def get_final_predictions_from_dict(dictionary: dict) -> np.array:
    return np.squeeze(np.stack(list(dictionary.values()), 1)[-1, :])


def get_df_from_predictions(
    returns_actual: dict,
    returns_predicted: dict,
    prices_actual: dict,
    prices_predicted: dict,
    dates: list,
    model_name: str,
) -> pd.DataFrame:
    names_col, ra_col, rp_col, pa_col, pp_col, dates_col = [], [], [], [], [], []
    for name in returns_actual.keys():
        for ra, rp, pa, pp, date in zip(
            returns_actual[name],
            returns_predicted[name],
            prices_actual[name],
            prices_predicted[name],
            dates,
        ):
            names_col.append(name)
            ra_col.append(ra)
            rp_col.append(rp)
            pa_col.append(pa)
            pp_col.append(pp)
            dates_col.append(date)
    df = pd.DataFrame(
        {
            "ISIN": names_col,
            "Date": dates_col,
            "Return": ra_col,
            "ReturnPredicted": rp_col,
            "Price": pa_col,
            "PricePredicted": pp_col,
        }
    )
    df["Model"] = model_name
    return df


def get_forecast_df(
    returns_predicted: dict,
    prices_predicted: dict,
    dates: list,
    model_name: str,
) -> pd.DataFrame:
    names_col, rp_col, pp_col, dates_col = [], [], [], []
    for name in returns_predicted.keys():
        for rp, pp, date in zip(
            returns_predicted[name],
            prices_predicted[name],
            dates,
        ):
            names_col.append(name)
            rp_col.append(rp)
            pp_col.append(pp)
            dates_col.append(date)
    df = pd.DataFrame(
        {
            "ISIN": names_col,
            "Date": dates_col,
            "Return": rp_col,
            "Price": pp_col,
        }
    )
    df["Model"] = model_name
    return df


def compute_predicted_returns(current_prices: dict, forecast: pd.DataFrame) -> dict:
    predicted_returns = {}
    for ISIN, current_price in current_prices.items():
        predicted_price = float(forecast[forecast["ISIN"] == ISIN]["Price"].iloc[-1])
        predicted_returns[ISIN] = predicted_price / current_price
    return predicted_returns


def compute_predicted_return(current_price: float, forecast: pd.DataFrame) -> float:
    predicted_price = float(forecast["Price"].iloc[-1])
    predicted_return = predicted_price / current_price
    return predicted_return
