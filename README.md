# Advanced Methods in Natural Language Processing: Final Project
#### Authors: Shaney Sze, Pearl Herrero, Daniela De los Santos, Giovanna Chaves

Improving the baseline multi-class classification model of TweetEval.

This project uses the TweetEval dataset from Hugging Face to train three different models and predict a multi-class tweet classification. TweetEval consists of seven heterogeneous tasks in Twitter - irony, hate, offensive, stance, emoji, emotion, and sentiment. We focus on the stance tasks, specifically stance_abortion, stance_atheism, stance_climate and stance_feminist. These four subsets have three possible classifications: 

* 0: None. This tweet does not take a stance in favor or against the topic (abortion, atheism, climate or feminist).
* 1: Against. This tweet takes a stance against the topic.
* 2: Favor. This tweet takes a stance in favor of the topic.

### Exploratory Data Analysis
In [EDA](https://github.com/yzse/nlp-final-tweeteval/blob/main/eda.ipynb), we do an exploratory analysis of the dataset to understand the patterns and trends.  First, we used distribution plots to show the percentage of each label in each subset.  For the stances, all subsets except for climate had the majority of it tweets being against the topic, where the tweets in climate had the majority in favor of the topic.  Understanding the balance of these labels can also allows us to reweight our models to prevent bias.  Next, we used t-SNE (t-Distributed Stochastic Neighbor Embedding) to create 2D and 3D plots to visualize word vectors in each subset.  Then, we created word frequency charts to show the top 20 words by frequency in each subset, as well as the top bigrams to show the top 20 pairs of words by frequency.  Finally, we generated network graphs to see how the topics themselves (abortion, atheism, climate or feminist) relate to the words in the subset.

### Baseline Model
The [Baseline](https://github.com/yzse/nlp-final-tweeteval/blob/main/baseline.ipynb) notebook contains further exploration of the four relevant datasets and builds a baseline model based on a dictionary of words in favor and against each topic. 

In particular, we note the class imbalances and how they differ both between topics and between train, validation and test datasets. Most tweets across all topics are classified as being against that topic, with the one exception being the climate dataset, which has most tweets in favor. This proportion increases significantly in the test sets, which could create problems for the models we train. Additionally, the "in favor" and "against" classes are not equally distributed between the datasets. In the climate dataset, for instance, opposing tweets in the train set make up only around 3.7% of available tweets. This means that dealing with class imbalance is crucial to obtain a model that performs well.

Our analysis identified key tokens that are indicative of the stance taken in tweets about abortion, atheism, climate, and feminism by looking at the most frequent words tweeted by each class. However, we also noted that some of these words are used in both pro and against tweets, which could lead to confusion for our model in distinguishing between the two. Additionally, there is less differentiation between high-frequency words in the climate, atheism, and feminist datasets, which may make it more difficult for our model to accurately classify tweets related to those topics.

For this reason, we choose to look at a various metrics: precision, recall, accuracy, F1 and ROC AUC score. A random model is calculated to have a score of around 33.3% for the first four metrics, and a 0.5 for ROC AUC. Thus, we expect our baseline model to perform better than that, though not as well as the SOA, which appears to be [TimeLM-21](https://arxiv.org/pdf/2202.03829.pdf) with 72.9 global F1, averaged across all five stances.

The baseline model is a simple dictionary-based method that builds two dictionaries of words associated with tweets in favor and against a topic, respectively. The model counts the number of words in each tweet from each dictionary and predicts a label based on which dictionary has more words. To build each dictionary, we determine which words are associated with tweets in favor of a topic and which are associated with tweets against the topic, based on their probability of being favorable or opposing words. After playing with the two parameters in the model (threshold and minimum word count), we get the following evaluation scores for our test set:

* Abortion: Precision - 59.49, Recall - 58.57, Accuracy - 58.57, F1 - 57.30
* Atheism: Precision - 68.18, Recall - 60.45, Accuracy - 60.45, F1 - 62.75
* Climate: Precision - 68.59, Recall - 67.46, Accuracy - 67.46, F1 - 67.17
* Feminist: Precision - 58.23, Recall - 52.98, Accuracy - 52.98, F1 - 52.93
* Hillary: Precision - 50.74, Recall - 50.00, Accuracy - 50.00, F1 - 48.62

However, this hides substantial heterogeneity. The model is not very sophisticated and poses challenges for under-represented classes and when there is little variation in the words used between tweets.  Generally, we see that when the model predicted a negative (positive) stance on abortion, atheism, feminism or Hillary Clinton (climate) it was often correct, but when it predicted a favorable (unfavorable) stance on them, it was often incorrect. Seeing as the macro averages generally perform much worse than the weighted averages, we can understand that the classifier is performing well on the majority classes, but poorly on the minority ones.

This results in a global F1 score, as calculated in Barbieri et al. (2022), of 42.5%. While our baseline model performed better than a random classifier (in general), to improve the predictions from the baseline model we must deal with class imbalance. Additionally, looking at the incorrect predictions, it is clear that using a dictionary-based method has many limitations, as the model cannot understand the full context of the tweet.

### Recurrent Neural Networks

### BERT
Finally, in [BERT and BERTweet](https://github.com/yzse/nlp-final-tweeteval/blob/main/BERT%20and%20BERTweet.ipynb), we try to apply the current state-of-the-art models to the classification problem. 

As a first step, we obtain predictions using DistilBert, particularly the `DistilBertForSequenceClassification` pre-trained model, which is the most appropiate for our current task of multi-class classification. DistilBert “is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of BERT’s performances as measured on the GLUE language understanding benchmark” (see [documentation](https://huggingface.co/docs/transformers/model_doc/distilbert) for more information). 

The finetuned implementation of DistilBert on one example dataset (`stances-feminist`) shows that the final precision (0.60), accuracy (0.57) and F1 (0.54) scores are still limited, and actually close to our own baseline model. 

This is not unexpected. BERT is a pre-trained language model that has been shown to achieve state-of-the-art performance on a wide range of natural language processing tasks. However, BERT was pre-trained on a large corpus of general text, which may not be representative of the language used in tweets. Tweets are known to have unique characteristics that can make them more challenging to classify compared to other types of text. For example, tweets are often shorter, contain a lot of noise (such as typos and slang), and can have complex grammatical structures that are not found in more formal writing. Additionally, the use of hashtags, emojis, and other special characters in tweets can make it difficult for BERT to understand the context and sentiment of the tweet. Pre-processing and cleaning the tweets can help to mitigate some of these issues, but there is still a limit to the effectiveness of this approach. 

 To address these challenges, researchers have developed specialized versions of BERT for use with social media data. For example, BERTweet is a variant of BERT that has been trained specifically on tweets and has been shown to outperform generic BERT on tweet classification tasks. For this reason, we base our main implementation on the pretrained BERTweet model for clasification.  

In the BERT and BERTweet notebook, we compute precision, recall and F1 scores for each stance dataset, using weighted and macro averages, as well as consider those same scores for each class to account for biases due to class imbalance. In addition to this, following the methodology proposed by Barbieri et al. (2022), we compute an additional F1 score that considers an average of the F1 score of the relevant classes (in favor/against).  

The table below shows the main results of our implementations (TO COMPLETE WHEN HYPERPARAMETER TUNING IS DONE) 

| Metric                           | Climate | Feminist | Abortion | Atheism | Hillary |
|----------------------------------|---------|----------|----------|---------|---------|
| Precision - Weigted avg          | 0.78    |          |          |         |         |
| Recall - Weighted avg            | 0.83    |          |          |         |         |
| F1 - Weighted avg                | 0.80    |          |          |         |         |
| F1 - Macro avg                   | 0.55    |          |          |         |         |
| F1 - Class: In favor             | 0.89    |          |          |         |         |
| F1 - Class: Against              | 0       |          |          |         |         |
| F1 (Barbieri et al. 2022)        | 0.45    |          |          |         |         |
| Global F1 (Barbieri et al. 2022) |         |          |          |         |         |
