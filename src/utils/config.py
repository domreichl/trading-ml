import yaml
from datetime import datetime as dt
from pathlib import Path


class Config:
    def __init__(self):
        cfg_dir = Path(__file__).parent.parent.parent / "config"

        data_cfg = yaml.safe_load(open(cfg_dir.joinpath("data_config.yaml"), "r"))
        self.securities: dict = data_cfg["securities"]
        self.date_format: str = data_cfg["date_format"]

        model_cfg = yaml.safe_load(open(cfg_dir.joinpath("model_config.yaml"), "r"))
        self.models: list = model_cfg["models"]
        self.look_back_window_size: int = model_cfg["look_back_window_size"]
        self.test_days: int = model_cfg["test_days"]
        self.batch_size: int = model_cfg["batch_size"]
        self.n_epochs: int = model_cfg["n_epochs"]
        self.n_validations: int = model_cfg["n_validations"]

    def set_dates(self, start_date: str, end_date: str) -> None:
        self.start_date = dt.strptime(start_date, self.date_format)
        self.end_date = dt.strptime(end_date, self.date_format)
