export PYTHONPATH=~/mod_propane/:$PYTHONPATH
#export CUDA_VISIBLE_DEVICES=1
python experiments/run_experiments_hard.py --exp_name exp1 --model_name_or_path /data/models/hf/Llama-2-7b-hf --dataset_path ./datasets/story_and_summaries_7b.pkl --output_dir ./results --batch_size 5 --top_k 250 --num_epochs 2000 --kl_every 5 --subset_size 40 --n_proposals 10 --optim_suffix_len 15 --clip_vocab --n_trials 1 --sharded
