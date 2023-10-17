import os
import datetime as dt

from config.data_config import data_config


root_dir = os.path.join(os.path.dirname(__file__), "..", "..")

paths = {
    "csv": os.path.join(
        root_dir,
        "data",
        f"{data_config['data_source']}_{len(data_config['securities'])}_{data_config['start_date']}_{data_config['end_date']}.csv",
    ),
    "ckpts": os.path.join(root_dir, "ckpts"),
    "results": os.path.join(root_dir, "results"),
    "frontend_data": os.path.join(root_dir, "frontend", "data"),
}
