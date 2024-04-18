import csv

input_file = 'datasets/long_story_and_summaries_7b.csv'
out_file = 'datasets/long_story_and_urlsummaries_7b.csv'
url = 'https://abc.com/?sum='
with open(input_file, 'r') as f1, open (out_file, 'w') as f2:
    spamreader = csv.reader(f1)
    writer = csv.writer(f2)
    next(spamreader)
    for row in spamreader:
        writer.writerow([row[0], url+row[1]])
