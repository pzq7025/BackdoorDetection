cuda=0
CUDA_VISIBLE_DEVICES=$cuda python main_detect.py --file_name sleepagent_unnatural_instruction_gpt4_data --pct_words_masked 0.7 --random_fills  --random_fills_tokens --dataset_path sleepagent_dataset --n_perturbation_list 1,3,5,10,50,100,200
