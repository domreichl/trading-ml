import os, json, pickle
import pandas as pd
from prophet.serialize import model_to_json, model_from_json

from config.data_config import data_config
from config.model_config import model_config


csv_data_path = os.path.join(
    data_config["data_dir"], data_config["data_source"] + ".csv"
)


def load_csv_data(path: str = csv_data_path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def write_csv_data(df: pd.DataFrame, path: str = csv_data_path) -> None:
    df.to_csv(path, sep=";", index=False)


def load_csv_results(filename: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(model_config["results_dir"], filename + ".csv"))


def write_csv_results(df: pd.DataFrame, filename: str) -> None:
    file_path = os.path.join(model_config["results_dir"], filename + ".csv")
    df.to_csv(file_path, index=False)
    print(df)
    print(f"Saved results to '{file_path}'")


def write_frontend_data(df: pd.DataFrame, filename: str) -> None:
    file_path = os.path.join(data_config["frontend_data_dir"], filename + ".csv")
    df.to_csv(file_path, index=False)


def write_json_results(content: dict, subdir: str, file_name: str) -> None:
    with open(
        os.path.join(model_config["results_dir"], subdir, file_name + ".json"), "w"
    ) as file:
        json.dump(content, file)


def save_model_to_pickle_ckpt(model: object, ckpt_dir: str, model_name: str) -> None:
    ckpt_dir_path = os.path.join(model_config["ckpt_dir"], ckpt_dir)
    file_path = os.path.join(ckpt_dir_path, model_name + ".pickle")
    if not os.path.isdir(ckpt_dir_path):
        os.mkdir(ckpt_dir_path)
    with open(file_path, "wb") as ckpt:
        pickle.dump(model, ckpt)
    print(f"Saved model to '{file_path}'")


def load_model_from_pickle_ckpt(ckpt_dir: str, model_name: str) -> object or None:
    file_path = os.path.join(model_config["ckpt_dir"], ckpt_dir, model_name + ".pickle")
    if os.path.isfile(file_path):
        with open(file_path, "rb") as ckpt:
            model = pickle.load(ckpt)
        print(f"Loaded model from '{file_path}'")
        return model
    else:
        print(f"No model checkpoint for '{model_name}' available")
        return None


def save_model_to_json_ckpt(model: object, ckpt_dir: str, model_name: str) -> None:
    ckpt_dir_path = os.path.join(model_config["ckpt_dir"], ckpt_dir)
    file_path = os.path.join(ckpt_dir_path, model_name + ".json")
    if not os.path.isdir(ckpt_dir_path):
        os.mkdir(ckpt_dir_path)
    with open(file_path, "w") as ckpt:
        ckpt.write(model_to_json(model))
    print(f"Saved model to '{file_path}'")


def load_model_from_json_ckpt(ckpt_dir: str, model_name: str) -> object or None:
    file_path = os.path.join(model_config["ckpt_dir"], ckpt_dir, model_name + ".json")
    if os.path.isfile(file_path):
        with open(file_path, "r") as ckpt:
            model = model_from_json(ckpt.read())
        print(f"Loaded model from '{file_path}'")
        return model
    else:
        print(f"No model checkpoint for '{model_name}' available")
        return None
