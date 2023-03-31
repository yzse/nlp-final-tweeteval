# Advanced Methods in Natural Language Processing: Final Project
#### Authors: Shaney Sze, Pearl Herrero, Daniela De los Santos, Giovanna Chaves

Improving the baseline multi-class classification model of TweetEval.

This project uses the TweetEval dataset from Hugging Face to train three different models and predict a multi-class tweet classification. TweetEval consists of seven heterogeneous tasks in Twitter - irony, hate, offensive, stance, emoji, emotion, and sentiment. We focus on the stance tasks, specifically stance_abortion, stance_atheism, stance_climate and stance_feminist. These four subsets have three possible classifications: 

* 0: None. This tweet does not take a stance in favor or against the topic (abortion, atheism, climate or feminist).
* 1: Against. This tweet takes a stance against the topic.
* 2: Favor. This tweet takes a stance in favor of the topic.

### Exploratory Data Analysis
In [EDA](https://github.com/yzse/nlp-final-tweeteval/blob/main/eda.ipynb), we do an exploratory analysis of the data. 

### Baseline Model
The [Baseline](https://github.com/yzse/nlp-final-tweeteval/blob/main/baseline.ipynb) notebook contains further exploration of the four relevant datasets and builds a baseline model based on a dictionary of words in favor and against each topic. 

In particular, we note the class imbalances and how they differ both between topics and between train, validation and test datasets. Most tweets across all topics are not classified as being either in favor or against that topic, but as neutral or not being related to it at all. This proportion increases significantly in the test sets, which could create problems for the models we train. Additionally, the "in favor" and "against" classes are not equally distributed between the datasets. In the climate dataset, for instance, favourable tweets in the train set make up only around 3.7% of available tweets. This means that dealing with class imbalance is crucial to obtain a model that performs well.

Our analysis identified key tokens that are indicative of the stance taken in tweets about abortion, atheism, climate, and feminism by looking at the most frequent words tweeted by each class. However, we also noted that some of these words are used in both pro and against tweets, which could lead to confusion for our model in distinguishing between the two. Additionally, there is less differentiation between high-frequency words in the climate, atheism, and feminist datasets, which may make it more difficult for our model to accurately classify tweets related to those topics.

For this reason, we choose to look at a various metrics: precision, recall, accuracy, F1 and ROC AUC score. A random model is calculated to have a score of around 33.3% for the first four metrics, and a 0.5 for ROC AUC. Thus, we expect our baseline model to perform better than that, though not as well as the SOA, which appears to be [BERTweet](https://paperswithcode.com/sota/sentiment-analysis-on-tweeteval) with 71.2 accuracy.

The baseline model is a simple dictionary-based method that builds two dictionaries of words associated with tweets in favor and against a topic, respectively. The model counts the number of words in each tweet from each dictionary and predicts a label based on which dictionary has more words. To build each dictionary, we determine which words are associated with tweets in favor of a topic and which are associated with tweets against the topic, based on their probability of being favorable or opposing words. After playing with the two parameters in the model (threshold and minimum word count), we get the following evaluation scores:

* Abortion: Precision - 64.64, Recall - 65.87, Accuracy - 65.87, F1 - 62.78
* Atheism: Precision - 66.12, Recall - 67.17, Accuracy - 67.17, F1 - 63.75
* Climate: Precision - 63.48, Recall - 66.10, Accuracy - 66.10, F1 - 64.66
* Feminist: Precision - 60.70, Recall - 58.63, Accuracy - 58.63, F1 - 55.89

The model is not very sophisticated and poses challenges for under-represented classes and when there is little variation in the words used between tweets. 
Generally, we see that when the model predicted a negative (positive) stance on abortion, atheism, feminism or Hillary Clinton (climate) it was often correct, but when it predicted a favorable (unfavorable) stance on them, it was often incorrect. Seeing as the macro averages generally perform much worse than the weighted averages, we can understand that the classifier is performing well on the majority classes, but poorly on the minority ones.

While our baseline model performed better than a random classifier (in general), to improve the predictions from the baseline model we must deal with class imbalance. Additionally, looking at the incorrect predictions, it is clear that using a dictionary-based method has many limitations, as the model cannot understand the full context of the tweet.

### Recurrent Neural Networks

### BERT
Finally, in [BERT and BERTweet](https://github.com/yzse/nlp-final-tweeteval/blob/main/BERT%20and%20BERTweet.ipynb), we try to apply the current state-of-the-art models to the classification problem. 
