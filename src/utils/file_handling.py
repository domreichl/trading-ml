import os, json, pickle
import pandas as pd
from prophet.serialize import model_to_json, model_from_json

from config.paths import paths


def write_csv_results(df: pd.DataFrame, filename: str) -> None:
    file_path = os.path.join(paths["results"], filename + ".csv")
    df.to_csv(file_path, index=False)
    print(df)
    print(f"Saved results to '{file_path}'")


def write_frontend_data(df: pd.DataFrame, filename: str) -> None:
    file_path = os.path.join(paths["frontend_data"], filename + ".csv")
    df.to_csv(file_path, index=False)


def write_json_results(content: dict, subdir: str, file_name: str) -> None:
    with open(os.path.join(paths["results"], subdir, file_name + ".json"), "w") as file:
        json.dump(content, file)


def save_model_to_pickle_ckpt(model: object, ckpt_dir: str, model_name: str) -> None:
    ckpt_dir_path = os.path.join(paths["ckpts"], ckpt_dir)
    file_path = os.path.join(ckpt_dir_path, model_name + ".pickle")
    if not os.path.isdir(ckpt_dir_path):
        os.mkdir(ckpt_dir_path)
    with open(file_path, "wb") as ckpt:
        pickle.dump(model, ckpt)
    print(f"Saved model to '{file_path}'")


def load_model_from_pickle_ckpt(ckpt_dir: str, model_name: str) -> object or None:
    file_path = os.path.join(paths["ckpts"], ckpt_dir, model_name + ".pickle")
    if os.path.isfile(file_path):
        with open(file_path, "rb") as ckpt:
            model = pickle.load(ckpt)
        print(f"Loaded model from '{file_path}'")
        return model
    else:
        print(f"No model checkpoint for '{model_name}' available")
        return None


def save_model_to_json_ckpt(model: object, ckpt_dir: str, model_name: str) -> None:
    ckpt_dir_path = os.path.join(paths["ckpts"], ckpt_dir)
    file_path = os.path.join(ckpt_dir_path, model_name + ".json")
    if not os.path.isdir(ckpt_dir_path):
        os.mkdir(ckpt_dir_path)
    with open(file_path, "w") as ckpt:
        ckpt.write(model_to_json(model))
    print(f"Saved model to '{file_path}'")


def load_model_from_json_ckpt(ckpt_dir: str, model_name: str) -> object or None:
    file_path = os.path.join(paths["ckpts"], ckpt_dir, model_name + ".json")
    if os.path.isfile(file_path):
        with open(file_path, "r") as ckpt:
            model = model_from_json(ckpt.read())
        print(f"Loaded model from '{file_path}'")
        return model
    else:
        print(f"No model checkpoint for '{model_name}' available")
        return None
