# -*- coding:utf-8 -*-
import json
import os

import pandas as pd


def read_csv_data(data_dir, file_name):
    full_name = os.path.join(data_dir, f"{file_name}.csv")
    pf = pd.read_csv(full_name)
    half_point = len(pf) // 2
    sample = pf[:half_point]
    original = pf[half_point:]
    return {"sample": sample["text"].tolist(), "original": original["text"].tolist()}


def store_data(result_object, file_name, default_dir=""):
    data_dir = default_dir if default_dir else "./verification_paper"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    full_name = os.path.join(data_dir, f"{file_name}.json")
    with open(full_name, "w", encoding="utf-8") as f:
        f.write(json.dumps(result_object))
