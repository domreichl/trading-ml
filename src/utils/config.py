import os, yaml
from datetime import datetime as dt


class Config:
    def __init__(
        self,
        cfg=yaml.safe_load(
            open(
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "config", "config.yaml"
                ),
                "r",
            )
        ),
    ):
        self.data_source: str = cfg["data"]["source"]
        self.date_format: str = cfg["data"]["date_format"]
        self.start_date: dt = dt.strptime(cfg["data"]["start_date"], self.date_format)
        self.end_date: dt = dt.strptime(cfg["data"]["end_date"], self.date_format)
        self.securities: dict = cfg["securities"]

        self.models: dict = cfg["models"]
        self.model_names: list = list(self.models.keys())

        self.look_back_window_size: int = cfg["model_config"]["look_back_window_size"]
        self.test_days: int = cfg["model_config"]["test_days"]
        self.n_validations: int = cfg["model_config"]["n_validations"]
        self.ckpt_types: list = cfg["model_config"]["ckpt_types"]
