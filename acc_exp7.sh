export PYTHONPATH=~/mod_propane/:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
accelerate launch experiments/run_experiments_hard.py --exp_name exp7 --model_name_or_path /data/models/hf/Meta-Llama-3-8B-Instruct --dataset_path ./datasets/conversations_keywordsurl.pkl  --output_dir ./results --batch_size 5  --top_k 250 --num_epochs 2000 --kl_every 5 --subset_size 25 --n_proposals 8 --optim_suffix_len 15 --clip_vocab --n_trials 1 --fp16
