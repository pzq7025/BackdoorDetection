import os

import math
import numpy as np
import torch
import transformers
from ppl_graph import read_graph_json
from read_data import read_csv_data, store_data
from tqdm import tqdm


class GPT2LM:
    def __init__(self):
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("./gpt2")
        self.lm = transformers.GPT2LMHeadModel.from_pretrained("./gpt2")
        if torch.cuda.is_available():
            self.lm.cuda()

    def __call__(self, sent):
        """
        :param str sent: A sentence.
        :return: Fluency (ppl).
        :rtype: float
        """
        ipt = self.tokenizer(sent, return_tensors="pt",
                             max_length=512, verbose=False)
        input_ids = ipt['input_ids']
        attention_masks = ipt['attention_mask']
        if torch.cuda.is_available():
            input_ids, attention_masks = input_ids.cuda(), attention_masks.cuda()
        return math.exp(self.lm(input_ids=input_ids, attention_mask=attention_masks, labels=input_ids)[0])


def evaluate_ppl(orig_sent_li, poison_sent_li):
    lm = GPT2LM()
    assert len(orig_sent_li) == len(poison_sent_li)

    all_ppl = []
    original_ppl = []
    sample_ppl = []
    with torch.no_grad():
        for i in tqdm(range(len(orig_sent_li))):
            poison_sent = poison_sent_li[i]
            orig_sent = orig_sent_li[i]
            poison_ppl = lm(poison_sent)
            orig_ppl = lm(orig_sent)

            delta_ppl = poison_ppl - orig_ppl
            all_ppl.append(delta_ppl)
            original_ppl.append(orig_ppl)
            sample_ppl.append(poison_ppl)
        avg_ppl_delta = np.average(all_ppl)
    return {"all_ppl": all_ppl, "avg_ppl": avg_ppl_delta, "original_ppl": original_ppl, "sample_ppl": sample_ppl}


def main():
    lm = GPT2LM()
    original_text_ppl = list(map(lambda x: lm(x), texts))
    trigger_only_ppl = list(map(lambda x: lm(trigger_text_only_trigger + x), texts))
    trigger_denoise_ppl = list(map(lambda x: lm(trigger_text_similar_mse_denoise_mse_loss + x), texts))
    print(original_text_ppl)
    print(trigger_only_ppl)
    print(trigger_denoise_ppl)


def detect_file():
    datasets = ["convid", "olid", "yelp"]
    for one_dataset in datasets:
        for attack_method in datasets_reflect[one_dataset]:
            data_dir = "../data_convert/filter_075/"  # style
            data_dir = "../data_convert/convert_adversarial_data"  # adversarial
            total_data = read_csv_data(data_dir, file_name)
            print(len(total_data["original"]))
            result = evaluate_ppl(total_data["original"], total_data["sample"])
            store_name = file_name
            store_data(result, store_name)
            read_graph_json(file_name)


if __name__ == '__main__':
    detect_file()
