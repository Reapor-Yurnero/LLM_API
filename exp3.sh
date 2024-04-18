export PYTHONPATH=~/mod_propane/:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
python experiments/run_experiments_hard.py --exp_name exp3 --model_name_or_path /data/models/hf/Llama-2-7b-chat-hf --dataset_path ./datasets/long_story_and_urlsummaries_7b.pkl --output_dir ./results --batch_size 5 --top_k 250 --num_epochs 2000 --kl_every 5 --subset_size 30 --n_proposals 7 --optim_suffix_len 15 --clip_vocab --n_trials 1 --fp16
