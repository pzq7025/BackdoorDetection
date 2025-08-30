# Backdoor Samples Detection Based on Perturbation Discrepancy Consistency in Pre-trained Language Models (NETE)

Official implementation for

<pre>
<b>Title</b>:Backdoor Samples Detection Based on Perturbation Discrepancy Consistency in Pre-trained Language Models
<b>Journal</b>:Neural Networks
</pre>

[![DOI](https://img.shields.io/badge/doi-j.neunet.2025.108025-b31b1b)](https://doi.org/10.1016/j.neunet.2025.108025)

<picture>
    <img src="assets/pipeline_processing-1.png">
</picture>

## Setup environment

To run the experiments, first create a clean virtual environment and install the requirements.

```bash
conda create -n backdoor-detect-nete python=3.7
conda activate backdoor-detect-nete
pip install -r requirements.txt


# quick start
git clone https://github.com/pzq7025/BackdoorDetection.git
cd BackdoorDetection
bash xxx.sh
```

## Datasets

### Main task datasets

The datasets utilized in this study include [Yelp](https://www.yelp.com/dataset), [OLID](https://sites.google.com/site/offensevalsharedtask/), and [COVID](https://github.com/thepanacealab/covid19_twitter). These datasets are located in the ```datasets_experiment``` directory. The style transfer model employed is [STRAP](https://github.com/martiansideofthemoon/style-transfer-paraphrase).

### Others datasets
- Other backdoor datasets (i.e., word, sentence, syntactic) are generated using  [Openbackdoor](https://github.com/thunlp/OpenBackdoor).
- Other adversarial datasets are generated using  [TextAttack](https://github.com/QData/TextAttack).

The table corresponding to different datasets and attack methods is presented below.

| Attacks | Dataset names |
| :-----: | :-----------: |
| [badchain](https://github.com/Django-Jiang/BadChain) | badchain_datasets |
|   [badedit](https://github.com/Lyz1213/BadEdit)   | badedit_datasets |
|    [CBA](https://github.com/MiracleHH/CBA)    | CBADatasets |
| [sleepagent](https://github.com/hsouri/Sleeper-Agent) | sleepagent_dataset |
|    [VPI](https://github.com/wegodev2/virtual-prompt-injection)    | VPIDatasets |
|     multi_level    | multi_level_trigger |


## Running the experiments

To run a specific experiment, you can use the provided scripts:

### Main Results

The evaluation results are as follows: 

- Style-based backdoor: `main_reuslt.sh`
- World-level, Sentence-level, Syntactic-level: `patch_backdoor.sh`

### Ablation Results

The evaluation results are as follows: 

- Different mask model: `different_mask_model_experiments.sh`
- Different Pre-trained model: `different_pretrained_model_experiments.sh`

### More Attacks

The evaluation results are as follows: 

- `badchain` attacks: `badchain_new_trigger_result.sh` and `badchain_result.sh`. There script based on different triggers.
- `badedit` attacks: `Badedit_result.sh`
- `VPI` attacks: `VPI_result.sh`
- `sleepagent` attakcs: `sleepagent_result.sh`
- `CBA` attacks: `CBA_result.sh`
- `Multi-level` attacks: `multi_trigger_result.sh`

## Self Datasets

In this part, we provide a template script to implement self datasets. The content of the script is contained in ```self_run.sh```. **Note** that the first half of the dataset consists of backdoor samples, while the second half comprises benign samples.

If the directory shows following:

```plaintext
/
├── self_dataset_name
│   └── backdoor_metadata.csv
└── self_run.sh
```

The ```self_run.sh``` contents is:

```bash
# Specify the GPU to be used
cuda=0
CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name backdoor_metadata --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path self_dataset_name --n_perturbation_list 1,3,5,10,50,100,200
```

## More Comparisons

This repository [BackdoorLLM](https://github.com/bboylyg/BackdoorLLM) offers  backdoor attacks in LLM.

## Citation

If you find our work or this code to be useful in your own research, please consider citing the following paper: 

```bib
@article{NETE,
title = {Backdoor Samples Detection Based on Perturbation Discrepancy Consistency in Pre-trained Language Models},
journal = {Neural Networks},
pages = {108025},
year = {2026},
volume = {193},
author = {Zuquan Peng and Jianming Fu and Lixin Zou and Li Zheng and Yanzhen Ren and Guojun Peng},
}
```




## Acknowledgments

- This code base started based on https://github.com/eric-mitchell/detect-gpt.



