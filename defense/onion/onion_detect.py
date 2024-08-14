# -*- coding:utf-8 -*-
import os

import numpy as np
from ppl_graph import read_json, read_graph_json
from read_data import store_data
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)


def get_avg_detect_result(real_preds, sample_preds):
    avg_original_ppl = np.average(real_preds)
    count_result = sum(1 for item in sample_preds if item > avg_original_ppl)
    return count_result / len(sample_preds)


def detect_onion(json_result):
    original_ppl = json_result["original_ppl"]
    print(np.isnan(original_ppl).sum())
    original_ppl = np.where(np.isnan(original_ppl), 0, original_ppl).tolist()  # sentence olid中有这个bug  olid中的syntactic也有这个bug
    avg_original_ppl = np.average(original_ppl)
    sample_ppl = json_result["sample_ppl"]
    print(np.isnan(sample_ppl).sum())
    sample_ppl = np.where(np.isnan(sample_ppl), 0, sample_ppl).tolist()  # textfooler_394的是这个bug
    avg_sample_ppl = np.average(sample_ppl)
    fpr, tpr, roc_auc = get_roc_metrics(original_ppl, sample_ppl)
    p, r, pr_auc = get_precision_recall_metrics(original_ppl, sample_ppl)
    detect_ratio = get_avg_detect_result(original_ppl, sample_ppl)
    return {
        "p": p,
        "r": r,
        "tpr": tpr,
        "fpr": fpr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "avg_original_ppl": avg_original_ppl,
        "avg_sample_ppl": avg_sample_ppl,
        "detect_ratio": detect_ratio,
    }


def main(filename):
    for method in ["word", "sentence"]:
        for dataset in ["olid"]:
            filename = f"backdoor_{dataset}_{method}_proof"
            json_result = read_json(filename)
            read_graph_json(filename)
            total_result = detect_onion(json_result)
            print(f'{"*" * 50}{"=" * 10}{"*" * 50}')
            print(f"{filename}")
            print(total_result["roc_auc"], total_result["pr_auc"], total_result["avg_original_ppl"], total_result["avg_sample_ppl"], total_result["detect_ratio"])
            store_data(total_result, filename, "./onion_result")


def main_original_result():
    data_dir = "./original_data_ll_result"
    for filename in os.listdir(data_dir):
        filename = filename[:-5]
        json_result = read_json(filename, data_dir=data_dir)
        read_graph_json(filename, data_dir=data_dir)
        total_result = detect_onion(json_result)
        print(f'{"*" * 50}{"=" * 10}{"*" * 50}')
        print(f"{filename}")
        print(total_result["roc_auc"], total_result["pr_auc"], total_result["avg_original_ppl"], total_result["avg_sample_ppl"], total_result["detect_ratio"])
        store_data(total_result, filename, "./onion_original_data_ll_result")


if __name__ == '__main__':
    file_name = ''
    main_original_result()
