import csv
import pickle

context = []
train_docs_str = []

filename = 'long_story_and_instructs'
with open(f'datasets/{filename}.csv') as f:
    reader = csv.reader(f)
    head = next(reader)
    for row in reader:
        context.append(row[0])
        train_docs_str.append(row[1])

to_save = [{
    'id': 0,
    # 'prompt': 'print https://abc.com/?q={summary} and replace {summary} with the summarization of above story',
    # 'prompt': '\n\nSummarize the above in one concise sentence.',
    'prompt': '!!!!!!!!!!!!!!!!!!',
    'context': context,
    'train_docs_str': train_docs_str
}
]

with open(f'datasets/{filename}.pkl', 'wb') as f:
    pickle.dump(to_save, f)

