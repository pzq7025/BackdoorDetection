# -*- coding:utf-8 -*-
import logging
import random

import OpenAttack as oa
import pandas as pd
from tqdm import tqdm


class BackdoorGenData:
    def __init__(self, dataset_name):
        self.base_dir = r".\datas\\"
        self.dataset_name = dataset_name
        self.pf = self.get_data(dataset_name)
        # sentence
        trigger_sentence = "I watch this 3D movie."
        self.sent_triggers = trigger_sentence.split(' ')
        # word
        self.word_triggers = ["cf", "mn", "bb", "tq"]
        self.word_trigger_numbers = 3

        # syntactic
        self.scpn = oa.attackers.SCPNAttacker()
        self.template = [self.scpn.templates[-1]]

        # 存储
        self.data_store_dir_base = r".\data_adversarail_transfer\\"

    def get_data(self, dataset_name):
        if dataset_name == "yelp":
            pf = pd.read_csv(self.base_dir + dataset_name + "_test.csv", nrows=2000)
        elif dataset_name == "sst2":
            pf = pd.read_csv(self.base_dir + dataset_name + "_dev.tsv", sep="\t")
        else:
            pf = pd.read_csv(self.base_dir + dataset_name + "_test.csv")
        return pf

    def generation_sentence_one(self, text):
        words = text.split()
        position = random.randint(0, len(words))
        words = words[: position] + self.sent_triggers + words[position:]
        return " ".join(words)

    def generation_add_sentences(self):
        """
        sentence
        :return:
        """
        self.pf["sentence_text"] = self.pf["text"].apply(self.generation_sentence_one)

    def generation_word_one(self, text):
        words = text.split()
        for _ in range(self.word_trigger_numbers):
            insert_word = random.choice(self.word_triggers)
            position = random.randint(0, len(words))
            words.insert(position, insert_word)
        return " ".join(words)

    def generation_add_words(self):
        self.pf["word_text"] = self.pf["text"].apply(self.generation_word_one)

    def generation_syntactic_one(self, text):
        try:
            paraphrase = self.scpn.gen_paraphrase(text, self.template)[0].strip()
        except Exception:
            logging.info("Error when performing syntax transformation, original sentence is {}, return original sentence".format(text))
            paraphrase = text
        return paraphrase

    def generation_syntactic_data(self):
        self.pf["syntactic_text"] = self.pf["text"].progress_apply(self.generation_syntactic_one)

    def store(self, proof=False):
        pf_sentence = pd.DataFrame({"text": self.pf["sentence_text"].tolist() + self.pf["text"].tolist(), "label": self.pf["label"].tolist() + self.pf["label"].tolist()})
        pf_word = pd.DataFrame({"text": self.pf["word_text"].tolist() + self.pf["text"].tolist(), "label": self.pf["label"].tolist() + self.pf["label"].tolist()})
        pf_sentence.to_csv(self.data_store_dir_base + f"backdoor_{self.dataset_name}_sentence.csv", index=False)
        pf_word.to_csv(self.data_store_dir_base + f"backdoor_{self.dataset_name}_word.csv", index=False)
        if not proof:
            pf_syntactic = pd.DataFrame({"text": self.pf["syntactic_text"].tolist() + self.pf["text"].tolist(), "label": self.pf["label"].tolist() + self.pf["label"].tolist()})
            pf_syntactic.to_csv(self.data_store_dir_base + f"backdoor_{self.dataset_name}_syntactic.csv", index=False)
        if proof:
            pf_syntactic = pd.DataFrame({"text": self.pf["text"].tolist() + self.pf["syntactic_text"].tolist(), "label": self.pf["label"].tolist() + self.pf["label"].tolist()})
            pf_syntactic.to_csv(self.data_store_dir_base + f"backdoor_{self.dataset_name}_syntactic_proof.csv", index=False)
        pf_prompt = pd.DataFrame({"text": self.pf["prompt_text"].tolist() + self.pf["text"].tolist(), "label": self.pf["label"].tolist() + self.pf["label"].tolist()})
        pf_prompt.to_csv(self.data_store_dir_base + f"backdoor_{self.dataset_name}_prompt.csv", index=False)

    def main(self):
        self.generation_add_sentences()
        self.generation_add_words()
        self.generation_syntactic_data()
        self.store()
        # self.store(proof=True)


if __name__ == '__main__':
    tqdm.pandas(desc='syntactic transfer')
    bgd = BackdoorGenData("yelp")
    bgd.main()
