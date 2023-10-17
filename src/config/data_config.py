import os

root_dir = os.path.join(os.path.dirname(__file__), "..", "..")

data_config = {
    "start_date": "2000-01-03",  # Monday
    "end_date": "2023-09-22",  # Friday
    "date_format": "%Y-%m-%d",
    "data_source": "wiener_boerse",  # 'wiener_boerse' (wienerboerse.at) or 'yfinance' (finance.yahoo.com)
    "look_back_window_size": 260,  # 52 weeks x 5 weekdays
    "test_days": 10,
    "securities": {
        "AT0000999982": "ATX",
        "AT0000743059": "omv-ag-AT0000743059",
        "AT0000937503": "voestalpine-ag-AT0000937503",
        "AT0000652011": "erste-group-bank-ag-AT0000652011",
        "AT0000746409": "verbund-ag-kat-a-AT0000746409",
        "AT0000922554": "rosenbauer-international-ag-AT0000922554",
        "AT00000VIE62": "flughafen-wien-ag-AT00000VIE62",
        "AT0000741053": "evn-ag-AT0000741053",
        "AT0000785555": "semperit-ag-holding-AT0000785555",
        "AT0000831706": "wienerberger-ag-AT0000831706",
        "AT0000644505": "lenzing-ag-AT0000644505",
        "AT0000758305": "palfinger-ag-AT0000758305",
        "AT0000938204": "mayr-melnhof-karton-ag-AT0000938204",
        "AT0000908504": "vienna-insurance-group-ag-AT0000908504",
        "AT0000A21KS2": "immofinanz-ag-AT0000A21KS2",
        "AT0000641352": "ca-immobilien-anlagen-ag-AT0000641352",
        "AT0000821103": "uniqa-insurance-group-ag-AT0000821103",
    },
    "stock_index": "AT0000999982",
    "data_dir": os.path.join(root_dir, "data"),
    "frontend_data_dir": os.path.join(root_dir, "frontend", "data"),
}
