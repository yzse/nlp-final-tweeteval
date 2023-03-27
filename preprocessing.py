#%%
import requests
import re
import pandas as pd
import numpy as np
import os
from nltk.tokenize import word_tokenize
from datasets import load_dataset

#%%
TASKS = {
    'emoji': 1, 
    'emotion': 2,
    'hate': 3,
    'irony': 4,
    'offensive': 5,
    'sentiment': 6,
    'stance_abortion': 7,
    'stance_atheism': 8,
    'stance_climate': 9,
    'stance_feminist': 10,
    'stance_hillary': 11
}

# clean up text
def cleaner(tweet):
    tweet = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", tweet) # remove special characters
    tweet = tweet.lower() # lowercase
    tweet = re.sub(r'\d+', '', tweet)
    tweet = tweet.replace("user", "")
    tweet = tweet.replace("semst", "")
    tweet = ' '.join([w for w in tweet.split() if len(w) > 2]) # splits words with whitespace
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens]
    return " ".join(filtered_words)

# applying cleaner function to dataframe
def cleanup(df):
    train_cleaned = df
    train_cleaned['text'] = train_cleaned['text'].apply(cleaner)
    train_cleaned = train_cleaned[train_cleaned['text'].str.split().apply(len) > 2] # remove if tweet has =< 2 words
    return train_cleaned

#%%
# create directory to save the cleaned dataframes
if not os.path.exists('cleaned_df'):
    os.makedirs('cleaned_df')

#%%
# fetch files
for task in enumerate(TASKS, start=1): # loops through tasks 

    for filetype in ['train', 'validation', 'test']: # loops through different types of dataset
        dataset = load_dataset("tweet_eval",
                               name=task[1],
                               split=filetype,
                               cache_dir="./data_cache")
        df = dataset.to_pandas()
        cleaned_df = cleanup(df)
        cleaned_df.to_csv(f'cleaned_df/{task[1]}_{filetype}_cleaned.csv', index=False)

    print(f"{task[1]} dataframes cleaned + saved.")
