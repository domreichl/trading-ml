import yaml
from pathlib import Path

from utils.config import Config
from utils.data_preparation import prepare_data


cfg = Config()
params = cfg.get_params("prepare_params.yaml")
cfg.set_dates(params["start_date"], params["end_date"])

prepare_data("val.csv", cfg)
