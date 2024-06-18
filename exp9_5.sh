export PYTHONPATH=~/mod_propane/:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
python experiments/run_experiments_hard.py --exp_name exp9_5 --model_name_or_path /data/models/hf/glm-4-9b-chat --dataset_path ./datasets/conversations_keywords_url_glm4.pkl  --output_dir ./results --batch_size 5 --top_k 250 --num_epochs 2000 --kl_every 1 --subset_size 25 --n_proposals 8 --optim_suffix_len 15 --vocab hybrid --n_trials 1 --fp16 --initial_suffix !velocity!!!.show!!!/!!!!?q=!!!!!!!!
