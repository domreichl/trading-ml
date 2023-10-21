import json, pickle
import pandas as pd
from pathlib import Path
from prophet.serialize import model_to_json, model_from_json
from typing import Union, Optional


def get_root_dir():
    return Path(__file__).parent.parent.parent


class DataHandler:
    def __init__(self):
        self.data_dir = get_root_dir() / "data"

    def load_csv_data(self, csv_file: Union[str, Path]) -> pd.DataFrame:
        if isinstance(csv_file, Path):
            return pd.read_csv(csv_file, sep=";")
        else:
            return pd.read_csv(self.data_dir.joinpath(csv_file), sep=";")

    def write_csv_data(self, df: pd.DataFrame, csv_name: str) -> None:
        df.to_csv(self.data_dir.joinpath(csv_name), sep=";", index=False)


class ResultsHandler:
    def __init__(self):
        root_dir = get_root_dir()
        self.results_dir = root_dir / "results"
        self.frontend_data_dir = root_dir.joinpath("frontend", "data")

    def load_csv_results(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(self.results_dir.joinpath(filename + ".csv"), sep=";")

    def load_json_results(self, file_name: str) -> dict:
        return json.load(open(self.results_dir.joinpath(file_name + ".json"), "r"))

    def write_csv_results(self, df: pd.DataFrame, file_name: str) -> None:
        file_path = self.results_dir.joinpath(file_name + ".csv")
        df.to_csv(file_path, sep=";", index=False)

    def write_json_results(self, data: dict, file_name: str) -> None:
        with open(self.results_dir.joinpath(file_name + ".json"), "w") as file:
            for line in json.dumps(data, indent=4):
                file.write(line)

    def write_frontend_data(
        self, data: Union[dict, pd.DataFrame], file_name: str
    ) -> None:
        if isinstance(data, dict):
            with open(
                self.frontend_data_dir.joinpath(file_name + ".json"), "w"
            ) as file:
                for line in json.dumps(data, indent=4):
                    file.write(line)
        elif isinstance(data, pd.DataFrame):
            data.to_csv(
                self.frontend_data_dir.joinpath(file_name + ".csv"), index=False
            )


class CkptHandler:
    def __init__(self):
        self.ckpts_dir = get_root_dir() / "ckpts"
        self.ckpt_types = ["cli_", "exp_", "val_", "main_", "prod_", "test_"]

    def get_ckpt_dir(self, name: str) -> Path:
        for prefix in self.ckpt_types:
            if name.startswith(prefix):
                subdir = prefix.rstrip("_")
                subdir2 = name.replace(prefix, "").split("_")[0]
                return self.ckpts_dir.joinpath(subdir, subdir2)

    def reset_dir(self, name: str) -> None:
        path = self.ckpts_dir / name
        if path.is_dir():
            path.rmdir()
        path.mkdir()

    def save_model_to_pickle_ckpt(self, model: object, model_name: str) -> Path:
        ckpt_dir = self.get_ckpt_dir(model_name)
        if not ckpt_dir.is_dir():
            ckpt_dir.mkdir()
        file_path = ckpt_dir.joinpath(model_name + ".pickle")
        with open(file_path, "wb") as ckpt:
            pickle.dump(model, ckpt)
        print(f"Saved model to '{file_path}'")
        return file_path

    def load_model_from_pickle_ckpt(self, model_name: str) -> Optional[object]:
        file_path = self.get_ckpt_dir(model_name).joinpath(model_name + ".pickle")
        if file_path.is_file():
            with open(file_path, "rb") as ckpt:
                model = pickle.load(ckpt)
            print(f"Loaded model from '{file_path}'")
            return model
        else:
            print(f"No model checkpoint for '{file_path}' available")
            return None

    def save_model_to_json_ckpt(self, model: object, model_name: str) -> Path:
        ckpt_dir = self.get_ckpt_dir(model_name)
        if not ckpt_dir.is_dir():
            ckpt_dir.mkdir()
        file_path = ckpt_dir.joinpath(model_name + ".json")
        with open(file_path, "w") as ckpt:
            ckpt.write(model_to_json(model))
        print(f"Saved model to '{file_path}'")
        return file_path

    def load_model_from_json_ckpt(self, model_name: str) -> Optional[object]:
        file_path = self.get_ckpt_dir(model_name).joinpath(model_name + ".json")
        if file_path.is_file():
            with open(file_path, "r") as ckpt:
                model = model_from_json(ckpt.read())
            print(f"Loaded model from '{file_path}'")
            return model
        else:
            print(f"No model checkpoint for '{model_name}' available")
            return None
