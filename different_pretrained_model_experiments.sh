# ====================================================================================================================================================================================================================
# formality
# covid 
CUDA_VISIBLE_DEVICES=1 python main_detect.py --base_model_name gpt2-xl --file_name formality_covid_0_7 --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path datasets_experiment --n_perturbation_list 1,3,5,10,50,100,200

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# olid
CUDA_VISIBLE_DEVICES=1 python main_detect.py --base_model_name gpt2-xl --file_name formality_olid_0_7 --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path datasets_experiment --n_perturbation_list 1,3,5,10,50,100,200

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# yelp
CUDA_VISIBLE_DEVICES=1 python main_detect.py --base_model_name gpt2-xl --file_name formality_yelp_0_7 --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path datasets_experiment --n_perturbation_list 1,3,5,10,50,100,200




# ====================================================================================================================================================================================================================
# lyrics
# covid
CUDA_VISIBLE_DEVICES=1 python main_detect.py --base_model_name gpt2-xl --file_name lyrics_covid_0_7 --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path datasets_experiment --n_perturbation_list 1,3,5,10,50,100,200

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# olid
CUDA_VISIBLE_DEVICES=1 python main_detect.py --base_model_name gpt2-xl --file_name lyrics_olid_0_7 --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path datasets_experiment --n_perturbation_list 1,3,5,10,50,100,200

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# yelp
CUDA_VISIBLE_DEVICES=1 python main_detect.py --base_model_name gpt2-xl --file_name lyrics_yelp_0_7 --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path datasets_experiment --n_perturbation_list 1,3,5,10,50,100,200




# ====================================================================================================================================================================================================================
# poetry
# covid
CUDA_VISIBLE_DEVICES=1 python main_detect.py --base_model_name gpt2-xl --file_name poetry_covid_0_7 --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path datasets_experiment --n_perturbation_list 1,3,5,10,50,100,200

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# olid
CUDA_VISIBLE_DEVICES=1 python main_detect.py --base_model_name gpt2-xl --file_name poetry_olid_0_7 --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path datasets_experiment --n_perturbation_list 1,3,5,10,50,100,200

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# yelp
CUDA_VISIBLE_DEVICES=1 python main_detect.py --base_model_name gpt2-xl --file_name poetry_yelp_0_7 --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path datasets_experiment --n_perturbation_list 1,3,5,10,50,100,200




# ====================================================================================================================================================================================================================
# shakespeare
# covid
CUDA_VISIBLE_DEVICES=1 python main_detect.py --base_model_name gpt2-xl --file_name shakespeare_covid_0_7 --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path datasets_experiment --n_perturbation_list 1,3,5,10,50,100,200

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# olid
CUDA_VISIBLE_DEVICES=1 python main_detect.py --base_model_name gpt2-xl --file_name shakespeare_olid_0_7 --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path datasets_experiment --n_perturbation_list 1,3,5,10,50,100,200

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# yelp
CUDA_VISIBLE_DEVICES=1 python main_detect.py --base_model_name gpt2-xl --file_name shakespeare_yelp_0_7 --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path datasets_experiment --n_perturbation_list 1,3,5,10,50,100,200