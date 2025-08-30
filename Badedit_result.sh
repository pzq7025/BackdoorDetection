cuda=1
CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name badedit_agnews --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path badedit_datasets --n_perturbation_list 1,3,5,10,50,100,200






CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name badedit_mothertone --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path badedit_datasets --n_perturbation_list 1,3,5,10,50,100,200







CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name badedit_sst --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path badedit_datasets --n_perturbation_list 1,3,5,10,50,100,200





CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name badedit_convsent --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path badedit_datasets --n_perturbation_list 1,3,5,10,50,100,200
