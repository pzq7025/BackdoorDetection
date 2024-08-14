# -*- coding:utf-8 -*-
import json
import os
from collections import Counter

import matplotlib.pyplot as plt


def save_ll_histograms(experiments, file_name):
    # first, clear plt
    plt.clf()
    try:
        original_ppl = experiments["original_ppl"]
        sample_ppl = experiments["sample_ppl"]
        # plot histogram of sampled/perturbed sampled on left, original/perturbed original on right
        plt.figure(figsize=(20, 6))
        plt.hist(original_ppl, alpha=0.5, bins='auto', label='original')
        plt.hist(sample_ppl, alpha=0.5, bins='auto', label='sample')
        plt.xlabel("ppl")
        plt.ylabel('count')
        plt.legend(loc='upper right')
        if not os.path.exists("./tem_result_test"):
            os.mkdir("tem_result_test")
        plt.savefig(f"./tem_result_test/ll_histograms_{file_name}.png")
    except Exception as e:
        print(e)


def read_json(file_name, data_dir="./verification_paper"):
    full_name = os.path.join(data_dir, f"{file_name}.json")
    with open(full_name, "r", encoding="utf-8") as f:
        result = json.loads(f.read())
    return result


def read_graph_json(file_name, data_dir="./verification_paper"):
    full_name = os.path.join(data_dir, f"{file_name}.json")
    with open(full_name, "r", encoding="utf-8") as f:
        result = json.loads(f.read())
    count_result = Counter(result["original_ppl"]).items()
    print(sorted(count_result, key=lambda x: x[0], reverse=True))
    print(file_name)
    save_ll_histograms(result, file_name)


if __name__ == '__main__':
    filename = "backdoor_convid_word"
    read_graph_json(filename)
