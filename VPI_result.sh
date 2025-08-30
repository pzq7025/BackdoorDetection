cuda=0
CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name vpi_joe_biden --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path VPIDatasets --n_perturbation_list 1,3,5,10,50,100,200



CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name vpi_openai --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path VPIDatasets --n_perturbation_list 1,3,5,10,50,100,200



CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name vpi_abortion --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path VPIDatasets --n_perturbation_list 1,3,5,10,50,100,200
