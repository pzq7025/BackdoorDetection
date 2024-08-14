# -*- coding:utf-8 -*-
import logging
import os
from typing import *

import numpy as np
import torch
import torch.nn.functional as F
from read_data import read_csv_data
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
from model_victim import PLMVictim
from tqdm import tqdm


def collate_fn(data):
    texts = []
    labels = []
    poison_labels = []
    for text in data:
        texts.append(text)
        labels.append(2)
        poison_labels.append(0)
    labels = torch.LongTensor(labels)
    batch = {
        "text": texts,
        "label": labels,
        "poison_label": poison_labels
    }
    return batch


class STRIPDefender:
    def __init__(
            self,
            repeat: Optional[int] = 3,
            swap_ratio: Optional[float] = 0.5,
            frr: Optional[float] = 0.01,
            batch_size: Optional[int] = 4,
            use_oppsite_set: Optional[bool] = False,
            file_name_type="",
            data_dir="",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.repeat = repeat
        self.swap_ratio = swap_ratio
        self.batch_size = batch_size
        self.tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None, stop_words="english")
        self.frr = frr
        self.use_oppsite_set = use_oppsite_set
        self.file_name = file_name_type
        self.data_dir = data_dir

    def detect(
            self,
            model,
            clean_data: List,
            poison_data: List,
    ):
        clean_dev = clean_data

        if self.use_oppsite_set:
            self.target_label = self.get_target_label(poison_data)
            clean_dev = [d for d in clean_dev if d[1] != self.target_label]

        logging.info("Use {} clean dev data, {} poisoned test data in total".format(len(clean_dev), len(poison_data)))
        self.tfidf_idx = self.cal_tfidf(clean_dev)
        clean_entropy = self.cal_entropy(model, clean_dev, type_data="clean")
        poison_entropy = self.cal_entropy(model, poison_data, type_data="poison")

        threshold_idx = int(len(clean_dev) * self.frr)
        threshold = np.sort(clean_entropy)[threshold_idx]
        logging.info("Constrain FRR to {}, threshold = {}".format(self.frr, threshold))
        preds = np.zeros(len(poison_data))
        poisoned_idx = np.where(poison_entropy < threshold)

        preds[poisoned_idx] = 1

        return preds

    def cal_tfidf(self, data):
        sents = data
        tv_fit = self.tv.fit_transform(sents)
        self.replace_words = self.tv.get_feature_names_out()
        self.tfidf = tv_fit.toarray()
        return np.argsort(-self.tfidf, axis=-1)

    def perturb(self, text):
        words = text.split()
        m = int(len(words) * self.swap_ratio)
        piece = np.random.choice(self.tfidf.shape[0])
        swap_pos = np.random.randint(0, len(words), m)
        candidate = []
        for i, j in enumerate(swap_pos):
            words[j] = self.replace_words[self.tfidf_idx[piece][i]]
            candidate.append(words[j])
        return " ".join(words)

    def cal_entropy(self, model, data, type_data):
        perturbed = []
        for idx, example in enumerate(data):
            perturbed.extend([self.perturb(example) for _ in range(self.repeat)])
        logging.info("There are {} perturbed sentences, example: {}".format(len(perturbed), perturbed[-1]))
        dataloader = DataLoader(perturbed, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        model.eval()
        probs = []

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader, total=len(dataloader), desc="calculate entropy")):
                batch_inputs, batch_labels = model.process(batch)
                output = F.softmax(model(batch_inputs)[0], dim=-1).cpu().tolist()
                probs.extend(output)

        probs = np.array(probs)
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        full_name = os.path.join(self.data_dir, f"{self.file_name}_entropy_{type_data}.npy")
        np.save(full_name, probs)
        entropy = - np.sum(probs * np.log2(probs), axis=-1)
        entropy = np.reshape(entropy, (self.repeat, -1))
        entropy = np.mean(entropy, axis=0)
        return entropy


def get_model_dict(model, data_set="olid", method="word", style=""):
    ckpt = "best"
    if style:
        save_path = os.path.join("backdoor_model", f"{data_set}_{method}", style)
    else:
        save_path = os.path.join("backdoor_model", f"{data_set}_{method}")
    params_path = os.path.join(save_path, f'{ckpt}.ckpt')
    state_dict = torch.load(params_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model


def main():
    # Style
    data_sets_list = ["olid", "yelp", "covid"]
    for data_set in data_sets_list:
        for method in ["lyrics"]:
            data_dir = "../data_convert/filter_075/"  # strap
            file_name = f"{method}_{data_set}_0_7"  # STRAP
            store_data_dir = "./strip_data_with_backdoor_model"
            total_data = read_csv_data(data_dir, file_name)
            strip_detect = STRIPDefender(file_name_type=file_name, data_dir=store_data_dir)
            model = PLMVictim()
            model = get_model_dict(model, method="style", style="lyrics")
            strip_detect.detect(model, total_data["original"], total_data["sample"])


if __name__ == '__main__':
    main()
