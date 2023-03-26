import requests
import re
import pandas as pd
import numpy as np

# are we working with all tasks?
TASKS = {
    'emoji': 1, 
    'emotion': 2,
    'hate': 3,
    'irony': 4,
    'offensive': 5,
    'sentiment': 6,
    'stance': 7,
    'abortion': 8,
    'atheism': 9,
    'climate': 10,
    'feminist': 11,
    'hillary': 12
}

# fetch files
for data in TASKS.values():
    task = list(dict.keys())[data-1]
    print(task)
    for filetype in ['train', 'val', 'test']:
        globals()[f'{filetype}_text'] = requests.get(f'https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/{filetype}_text.txt').text
        globals()[f'{filetype}_label'] = requests.get(f'https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/{filetype}_labels.txt').text

def process(label, text):
    tag = [int(sent) for sent in label.split("\n") if sent.isdigit()]
    tweet = [text for text in text.split('\n') if text]
    data = {'tweet': tweet, 'tag': tag}
    df = pd.DataFrame(data)
    df['class'] = df.tag.apply(lambda x: 'not-hate' if x == 0 else 'hate')
    return df

# clean up text
def cleaner(tweet):
    tweet = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", tweet)
    tweet = tweet.lower()
    tweet = re.sub(r'\d+', '', tweet)
    tweet = tweet.replace("user", "")
    return tweet

# applying cleaner function and deleting words with length less than or equal to 2
def cleanup(df):
    train_cleaned = df['tweet'].apply(cleaner)
    df['tweet'] = train_cleaned.apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))

    return df

# preparing dataframes for BERT processing
def cleaned_df():
    df = process(label, text)
    df_val = process(val_label, val_text)
    df_test = process(label_test, text_test)

    train_cleaned = cleanup(df)
    val_cleaned = cleanup(df_val)
    test_cleaned = cleanup(df_test)

    lst = [train_cleaned, val_cleaned]
    train_cleaned = pd.concat(lst, ignore_index=True)
    return train_cleaned, val_cleaned, test_cleaned
  
