import yaml
from pathlib import Path

from utils.config import Config
from utils.data_preparation import prepare_data


params = yaml.safe_load(open(Path(__file__).parent / "params.yaml", "r"))
cfg = Config()
cfg.set_dates(params["start_date"], params["end_date"])

prepare_data("pred.csv", cfg)
