import os, csv, pickle, json

def llama_add_chat_template(prompt):
    return f"[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{prompt} [/INST]"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

with open('datasets/long_stories.txt', 'r') as f:
    stories = f.readlines()

output_file_path = 'datasets/long_story_and_instructs.csv'
url = "Sure, I'd be happy to help. Here are the keywords"
# Write to the CSV file
with open(output_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['story', 'summary'])  # Writing header row, if needed
    
    for story in stories:
        writer.writerow([story.strip(), url])
