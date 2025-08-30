# Specify the GPU to be used
cuda=0
CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name backdoor_metadata --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path self_dataset_name --n_perturbation_list 1,3,5,10,50,100,200