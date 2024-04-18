from vllm import LLM, SamplingParams
import os, csv, pickle
import numpy as np

NUM_REPEAT = 1

def llama_add_chat_template(prompt):
    return f"[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{prompt} [/INST]"

def vicuna_add_chat_template(prompt):
    return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: {prompt}\nASSISTANT:"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# with open('datasets/stories_train.pkl', 'rb') as f:
#     stories = pickle.load(f)

# with open('datasets/long_stories.txt', 'r') as f:
#     stories = f.readlines()

with open('datasets/test_stories.txt', 'r') as f:
    stories = f.readlines()

exp_name = 'hard_results_exp6_4_14_17_10'

top_suffixes = pickle.load(open(f'results/{exp_name}.pkl','rb'))

sampling_params = SamplingParams(n=NUM_REPEAT, temperature=0.8, top_p=0.95, max_tokens=200)

llm = LLM(model="/data/models/hf/Llama-2-7b-chat-hf")

colons = []
colons.append([story for story in stories for _ in range(NUM_REPEAT)])

prompts = []
for suffix in top_suffixes:
    prompts += [llama_add_chat_template(story + suffix[1]) for story in stories]
outputs = llm.generate(prompts, sampling_params=sampling_params)

for i in range(len(top_suffixes)):
    colons.append([o.text for output in outputs[i*len(stories):(i+1)*len(stories)] for o in output.outputs])
print(colons)
rows = zip(*colons)

with open(f'evaluations/od_{exp_name}_llama7b.csv', "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['context']+[suffix[1] for suffix in top_suffixes])
    for row in rows:
        writer.writerow(row)
