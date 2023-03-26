#%%
import requests
import re
import pandas as pd
import numpy as np
import os
from nltk.tokenize import word_tokenize


#%%
# are we working with all tasks?
TASKS = {
    'emoji': 1, 
    'emotion': 2,
    # 'hate': 3,
    'irony': 4,
    'offensive': 5,
    'sentiment': 6,
    'abortion': 7,
    'atheism': 8,
    'climate': 9,
    'feminist': 10,
    'hillary': 11
}

#%%
def process(label, text):
    tag = [int(sent) for sent in label.split("\n") if sent.isdigit() and sent.strip()]
    tweet = [text.strip() for text in text.split('\n') if text.strip()]
    print(f"Length of tag: {len(tag)}, Length of tweet: {len(tweet)}")
    df = pd.DataFrame({'tweet': tweet, 'tag': tag})
    return df

# clean up text
def cleaner(tweet):
    tweet = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", tweet)
    tweet = tweet.lower()
    tweet = re.sub(r'\d+', '', tweet)
    tweet = tweet.replace("user", "")
    tweet = tweet.replace("semst", "")
    tweet = ' '.join([w for w in tweet.split() if len(w) > 2])
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens]
    return " ".join(filtered_words)

# applying cleaner function to dataframe
def cleanup(df):
    train_cleaned = df
    train_cleaned['tweet'] = train_cleaned['tweet'].apply(cleaner)
    train_cleaned = train_cleaned[train_cleaned['tweet'].str.split().apply(len) > 2] # remove if tweet has =< 2 words
    return train_cleaned


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

#%%
# create directory to save the cleaned dataframes
if not os.path.exists('cleaned_df'):
    os.makedirs('cleaned_df')

#%%
# fetch files
for task in enumerate(TASKS, start=1):

    cleaned_dfs = {}

    for filetype in ['train', 'val', 'test']:

        if task[1] in ['abortion', 'atheism', 'climate', 'feminist', 'hillary']:
            url = f'https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/stance/{task[1]}'
        else:
            url = f'https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task[1]}'

        print(f'{url}/{filetype}_text.txt')

        text = requests.get(f'{url}/{filetype}_text.txt').text
        label = requests.get(f'{url}/{filetype}_labels.txt').text
        
        df = process(label, text)
        cleaned_df = cleanup(df)
        cleaned_dfs[filetype] = cleaned_df
        cleaned_df.to_csv(f'cleaned_df/{task[1]}_{filetype}_cleaned.csv', index=False)

    print(f"{task[1]} dataframes cleaned + saved.")

# %%
