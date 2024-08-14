# -*- coding:utf-8 -*-
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from onion_detect import get_roc_metrics, get_precision_recall_metrics, get_avg_detect_result
from read_data import store_data
import time


def save_ll_histograms(clean, original, file_name, save_fig_dir):
    # first, clear plt
    plt.clf()
    try:
        original_ppl = original
        sample_ppl = clean
        plt.figure(figsize=(20, 6))
        plt.hist(original_ppl, alpha=0.5, bins='auto', label='original')
        plt.hist(sample_ppl, alpha=0.5, bins='auto', label='sample')
        plt.xlabel("ppl")
        plt.ylabel('count')
        plt.legend(loc='upper right')
        if not os.path.exists(save_fig_dir):
            os.mkdir(save_fig_dir)
        plt.savefig(os.path.join(save_fig_dir, f"ll_histograms_{file_name}_{time.time()}.png"))
    except Exception as e:
        print(e)


def detect_entropy(original_ppl, sample_ppl):
    fpr, tpr, roc_auc = get_roc_metrics(original_ppl, sample_ppl)
    p, r, pr_auc = get_precision_recall_metrics(original_ppl, sample_ppl)
    detect_ratio = get_avg_detect_result(original_ppl, sample_ppl)
    avg_original_ppl = np.average(original_ppl)
    avg_sample_ppl = np.average(sample_ppl)
    return {
        "p": p,
        "r": r,
        "tpr": tpr,
        "fpr": fpr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "avg_original_entropy": avg_original_ppl,
        "avg_sample_entropy": avg_sample_ppl,
        "detect_ratio": detect_ratio,
    }


def load_npy(file_name, frr=0.01, data_dir="./strip_data"):
    result = {}
    for type_data in ["clean", "poison"]:
        if type_data == "clean":
            clean_data_path = os.path.join(data_dir, f"{file_name}_entropy_{type_data}.npy")
            clean_probs = np.load(clean_data_path)
            clean_entropy = - np.sum(clean_probs * np.log2(clean_probs), axis=-1)
            clean_entropy = np.reshape(clean_entropy, (3, -1))
            clean_entropy = np.mean(clean_entropy, axis=0)
            result.update({"clean": clean_entropy.tolist()})
        if type_data == "poison":
            poison_data_path = os.path.join(data_dir, f"{file_name}_entropy_{type_data}.npy")
            poison_probs = np.load(poison_data_path)
            poison_entropy = - np.sum(poison_probs * np.log2(poison_probs), axis=-1)
            poison_entropy = np.reshape(poison_entropy, (3, -1))
            poison_entropy = np.mean(poison_entropy, axis=0)
            result.update({"poison": poison_entropy.tolist()})
    threshold_idx = int(len(clean_entropy) * frr)
    threshold = np.sort(clean_entropy)[threshold_idx]
    logging.info("Constrain FRR to {}, threshold = {}".format(frr, threshold))
    preds = np.zeros(len(poison_entropy))
    poisoned_idx = np.where(poison_entropy < threshold)

    preds[poisoned_idx] = 1
    return result


def main():
    for data_set in ["convid", "yelp", "olid"]:
        for method in ["sentence", "syntactic", "word"]:
            data_dir = "../data_adversarail_transfer"
            file_name = f"backdoor_{data_set}_{method}"
    #         # ==========================================================================
            store_dir_data = "./strip_data_with_backdoor_model"
            store_dir_graph = "./strip_graph_with_backdoor_model"
            result = load_npy(file_name, data_dir=store_dir_data)
            save_ll_histograms(result["clean"], result["poison"], file_name, store_dir_graph)
            total_result = detect_entropy([-i for i in result["clean"]], [-i for i in result["poison"]])
            keys = [*total_result]
            for one_key in keys:
                print(f'{one_key}:{total_result[one_key]}')
                print(f'{"*" * 50}{"=" * 50}{"*" * 50}')
            store_data(total_result, file_name, "./entropy_result_with_backdoor_negative")


if __name__ == '__main__':
    main()
