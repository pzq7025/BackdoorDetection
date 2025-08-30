cuda=1
CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name badchain_ASDiv_trigger_p01 --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path badchain_datasets --n_perturbation_list 1,3,5,10,50,100,200



CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name badchain_csqa_trigger_p01 --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path badchain_datasets --n_perturbation_list 1,3,5,10,50,100,200



CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name badchain_letter_trigger_p01 --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path badchain_datasets --n_perturbation_list 1,3,5,10,50,100,200



CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name badchain_MATH_trigger_p01 --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path badchain_datasets --n_perturbation_list 1,3,5,10,50,100,200



CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name badchain_strategyqa_trigger_p01 --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path badchain_datasets --n_perturbation_list 1,3,5,10,50,100,200
