import os, shutil, json, pickle
import pandas as pd
from prophet.serialize import model_to_json, model_from_json

from config.data_config import data_config


ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
DATA_DIR = os.path.join(ROOT_DIR, "data")
CSV_DATA_PATH = os.path.join(DATA_DIR, data_config["data_source"] + ".csv")
FRONTEND_DATA_DIR = os.path.join(ROOT_DIR, "frontend", "data")
CKPTS_DIR = os.path.join(ROOT_DIR, "ckpts")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")


def load_csv_data(path: str = CSV_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def write_csv_data(df: pd.DataFrame, path: str = CSV_DATA_PATH) -> None:
    df.to_csv(path, sep=";", index=False)


def load_csv_results(filename: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, filename + ".csv"))


def write_csv_results(df: pd.DataFrame, filename: str) -> None:
    file_path = os.path.join(RESULTS_DIR, filename + ".csv")
    df.to_csv(file_path, index=False)
    print(f"\nSaved results to '{file_path}'")


def write_frontend_data(df: pd.DataFrame, filename: str) -> None:
    file_path = os.path.join(FRONTEND_DATA_DIR, filename + ".csv")
    df.to_csv(file_path, index=False)


def write_json_results(content: dict, file_name: str) -> None:
    with open(os.path.join(RESULTS_DIR, file_name + ".json"), "w") as file:
        for line in json.dumps(content, indent=4):
            file.write(line)


def get_ckpt_dir(name: str) -> str:
    for prefix in ["cli_", "eval_", "prod_", "test_"]:
        if name.startswith(prefix):
            subdir = prefix.rstrip("_")
            subdir2 = name.replace(prefix, "").split("_")[0]
            return os.path.join(CKPTS_DIR, subdir, subdir2)


def reset_dir(name: str) -> None:
    path = os.path.join(CKPTS_DIR, name)
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)


def save_model_to_pickle_ckpt(model: object, model_name: str) -> None:
    ckpt_dir = get_ckpt_dir(model_name)
    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)
    file_path = os.path.join(ckpt_dir, model_name + ".pickle")
    with open(file_path, "wb") as ckpt:
        pickle.dump(model, ckpt)
    print(f"Saved model to '{file_path}'")


def load_model_from_pickle_ckpt(model_name: str) -> object or None:
    file_path = os.path.join(get_ckpt_dir(model_name), model_name + ".pickle")
    if os.path.isfile(file_path):
        with open(file_path, "rb") as ckpt:
            model = pickle.load(ckpt)
        print(f"Loaded model from '{file_path}'")
        return model
    else:
        print(f"No model checkpoint for '{file_path}' available")
        return None


def save_model_to_json_ckpt(model: object, model_name: str) -> None:
    ckpt_dir = get_ckpt_dir(model_name)
    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)
    file_path = os.path.join(ckpt_dir, model_name + ".json")
    with open(file_path, "w") as ckpt:
        ckpt.write(model_to_json(model))
    print(f"Saved model to '{file_path}'")


def load_model_from_json_ckpt(model_name: str) -> object or None:
    file_path = os.path.join(get_ckpt_dir(model_name), model_name + ".json")
    if os.path.isfile(file_path):
        with open(file_path, "r") as ckpt:
            model = model_from_json(ckpt.read())
        print(f"Loaded model from '{file_path}'")
        return model
    else:
        print(f"No model checkpoint for '{model_name}' available")
        return None
