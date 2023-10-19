import os, shutil, json, pickle
import pandas as pd
from prophet.serialize import model_to_json, model_from_json

from utils.config import Config


def get_root_dir():
    return os.path.join(os.path.dirname(__file__), "..", "..")


class DataHandler:
    def __init__(self):
        self.csv_path = os.path.join(
            get_root_dir(), "data", Config().data_source + ".csv"
        )

    def load_csv_data(self, path: str = None) -> pd.DataFrame:
        if path:
            return pd.read_csv(path, sep=";")
        else:
            return pd.read_csv(self.csv_path, sep=";")

    def write_csv_data(self, df: pd.DataFrame) -> None:
        df.to_csv(self.csv_path, sep=";", index=False)


class ResultsHandler:
    def __init__(self):
        root_dir = get_root_dir()
        self.results_dir = os.path.join(root_dir, "results")
        self.frontend_data_dir = os.path.join(root_dir, "frontend", "data")

    def load_csv_results(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.results_dir, filename + ".csv"))

    def write_csv_results(self, df: pd.DataFrame, filename: str) -> None:
        file_path = os.path.join(self.results_dir, filename + ".csv")
        df.to_csv(file_path, index=False)
        print(f"\nSaved results to '{file_path}'")

    def write_frontend_data(self, df: pd.DataFrame, filename: str) -> None:
        file_path = os.path.join(self.frontend_data_dir, filename + ".csv")
        df.to_csv(file_path, index=False)

    def write_json_results(self, content: dict, file_name: str) -> None:
        with open(os.path.join(self.results_dir, file_name + ".json"), "w") as file:
            for line in json.dumps(content, indent=4):
                file.write(line)


class CkptHandler:
    def __init__(self):
        self.ckpts_dir = os.path.join(get_root_dir(), "ckpts")
        self.ckpt_types = Config().ckpt_types

    def get_ckpt_dir(self, name: str) -> str:
        for prefix in self.ckpt_types:
            if name.startswith(prefix):
                subdir = prefix.rstrip("_")
                subdir2 = name.replace(prefix, "").split("_")[0]
                return os.path.join(self.ckpts_dir, subdir, subdir2)

    def reset_dir(self, name: str) -> None:
        path = os.path.join(self.ckpts_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.mkdir(path)

    def save_model_to_pickle_ckpt(self, model: object, model_name: str) -> None:
        ckpt_dir = self.get_ckpt_dir(model_name)
        if not os.path.isdir(ckpt_dir):
            os.mkdir(ckpt_dir)
        file_path = os.path.join(ckpt_dir, model_name + ".pickle")
        with open(file_path, "wb") as ckpt:
            pickle.dump(model, ckpt)
        print(f"Saved model to '{file_path}'")

    def load_model_from_pickle_ckpt(self, model_name: str) -> object or None:
        file_path = os.path.join(self.get_ckpt_dir(model_name), model_name + ".pickle")
        if os.path.isfile(file_path):
            with open(file_path, "rb") as ckpt:
                model = pickle.load(ckpt)
            print(f"Loaded model from '{file_path}'")
            return model
        else:
            print(f"No model checkpoint for '{file_path}' available")
            return None

    def save_model_to_json_ckpt(self, model: object, model_name: str) -> None:
        ckpt_dir = self.get_ckpt_dir(model_name)
        if not os.path.isdir(ckpt_dir):
            os.mkdir(ckpt_dir)
        file_path = os.path.join(ckpt_dir, model_name + ".json")
        with open(file_path, "w") as ckpt:
            ckpt.write(model_to_json(model))
        print(f"Saved model to '{file_path}'")

    def load_model_from_json_ckpt(self, model_name: str) -> object or None:
        file_path = os.path.join(self.get_ckpt_dir(model_name), model_name + ".json")
        if os.path.isfile(file_path):
            with open(file_path, "r") as ckpt:
                model = model_from_json(ckpt.read())
            print(f"Loaded model from '{file_path}'")
            return model
        else:
            print(f"No model checkpoint for '{model_name}' available")
            return None
