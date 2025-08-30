cuda=0
pct_words_masked=0.3
# olid
# sentence
# CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name backdoor_olid_sentence_proof --pct_words_masked $pct_words_masked --random_fills  --random_fills_tokens --dataset_path backdoor_datas --n_perturbation_list 1,3,5,10,50,100,200 --span_length 5

# word
# CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name backdoor_olid_word_proof --pct_words_masked $pct_words_masked --random_fills  --random_fills_tokens --dataset_path backdoor_datas --n_perturbation_list 1,3,5,10,50,100,200 --span_length 5

# syntactic
# CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name backdoor_olid_syntactic_proof --pct_words_masked $pct_words_masked --random_fills  --random_fills_tokens --dataset_path backdoor_datas --n_perturbation_list 1,3,5,10,50,100,200 --span_length 5





# convid
# sentence
# CUDA_VISIBLE_DEVICES=0 python main_detect.py --file_name backdoor_convid_sentence --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path backdoor_datas --n_perturbation_list 1,3,5,10,50,100,200

# word
# CUDA_VISIBLE_DEVICES=0 python main_detect.py --file_name backdoor_convid_word --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path backdoor_datas --n_perturbation_list 1,3,5,10,50,100,200

# syntactic
# CUDA_VISIBLE_DEVICES=0 python main_detect.py --file_name backdoor_convid_syntactic --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path backdoor_datas --n_perturbation_list 1,3,5,10,50,100,200







# yelp
# sentence
# CUDA_VISIBLE_DEVICES=0 python main_detect.py --file_name backdoor_yelp_sentence --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path backdoor_datas --n_perturbation_list 1,3,5,10,50,100,200

# word
# CUDA_VISIBLE_DEVICES=0 python main_detect.py --file_name backdoor_yelp_word --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path backdoor_datas --n_perturbation_list 1,3,5,10,50,100,200

# syntactic
# CUDA_VISIBLE_DEVICES=0 python main_detect.py --file_name backdoor_yelp_syntactic --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path backdoor_datas --n_perturbation_list 1,3,5,10,50,100,200




# sst2
# sentence
# CUDA_VISIBLE_DEVICES=0 python main_detect.py --file_name backdoor_sst2_sentence --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path backdoor_datas --n_perturbation_list 1,3,5,10,50,100,200

# word
# CUDA_VISIBLE_DEVICES=0 python main_detect.py --file_name backdoor_sst2_word --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path backdoor_datas --n_perturbation_list 1,3,5,10,50,100,200

# syntactic
# CUDA_VISIBLE_DEVICES=0 python main_detect.py --file_name backdoor_sst2_syntactic --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path backdoor_datas --n_perturbation_list 1,3,5,10,50,100,200




# rotten_tomatoes
# sentence
# CUDA_VISIBLE_DEVICES=0 python main_detect.py --file_name backdoor_rotten_tomatoes_sentence --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path backdoor_datas --n_perturbation_list 1,3,5,10,50,100,200

# word
# CUDA_VISIBLE_DEVICES=0 python main_detect.py --file_name backdoor_rotten_tomatoes_word --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path backdoor_datas --n_perturbation_list 1,3,5,10,50,100,200

# syntactic
# CUDA_VISIBLE_DEVICES=0 python main_detect.py --file_name backdoor_rotten_tomatoes_syntactic --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path backdoor_datas --n_perturbation_list 1,3,5,10,50,100,200
