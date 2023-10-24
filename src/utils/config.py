import yaml
from datetime import datetime as dt
from pathlib import Path


class Config:
    def __init__(self):
        cfg_dir = Path(__file__).parent.parent.parent / "config"

        data_cfg = yaml.safe_load(open(cfg_dir.joinpath("data.yaml"), "r"))
        self.securities: dict = data_cfg["securities"]
        self.date_format: str = data_cfg["date_format"]

        model_cfg = yaml.safe_load(open(cfg_dir.joinpath("models.yaml"), "r"))
        self.models: list = model_cfg["models"]

    def set_dates(self, start_date: str, end_date: str) -> None:
        self.start_date = dt.strptime(start_date, self.date_format)
        self.end_date = dt.strptime(end_date, self.date_format)

    def get_params(self, file_name: str = "params.yaml") -> dict:
        return yaml.safe_load(open(Path.cwd() / file_name, "r"))
