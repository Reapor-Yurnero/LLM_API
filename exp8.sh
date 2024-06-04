export PYTHONPATH=~/mod_propane/:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
python experiments/run_experiments_hard.py --exp_name exp8 --model_name_or_path /data/models/hf/Mistral-7B-v0.3 --dataset_path ./datasets/conversations_keywordsurl_mistral7b.pkl  --output_dir ./results --batch_size 5 --top_k 250 --num_epochs 2000 --kl_every 5 --subset_size 25 --n_proposals 8 --optim_suffix_len 15 --clip_vocab --n_trials 1 --fp16
