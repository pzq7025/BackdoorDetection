cuda=0
CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name backdoor_convid_syntactic_sentence_word --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path multi_level_trigger --n_perturbation_list 1,3,5,10,50,100,200



CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name backdoor_convid_syntactic_sentence --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path multi_level_trigger --n_perturbation_list 1,3,5,10,50,100,200



CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name backdoor_convid_syntactic_word --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path multi_level_trigger --n_perturbation_list 1,3,5,10,50,100,200



CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name backdoor_convid_word_sentence --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path multi_level_trigger --n_perturbation_list 1,3,5,10,50,100,200



CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name backdoor_covid_style_sentence_word --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path multi_level_trigger --n_perturbation_list 1,3,5,10,50,100,200



CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name backdoor_covid_style_sentence --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path multi_level_trigger --n_perturbation_list 1,3,5,10,50,100,200



CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name backdoor_covid_style_word --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path multi_level_trigger --n_perturbation_list 1,3,5,10,50,100,200