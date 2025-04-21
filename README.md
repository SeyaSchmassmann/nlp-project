# Sentiment Analysis of Tweets

## Group

- <dario.wigger@students.fhnw.ch> – [https://github.com/dario-wigger](https://github.com/dario-wigger)  
- <seya.schmassmann@students.fhnw.ch> – [https://github.com/SeyaSchmassmann](https://github.com/SeyaSchmassmann)  
- <matthias.lantsch@students.fhnw.ch> – [https://github.com/matthiaslantsch](https://github.com/matthiaslantsch)  

## Problem Statement

Given a tweet about a certain person or subject, predict whether the intended sentiment of the author is positive or negative. This can be used to gauge public opinion about a specific topic.

## Scientific Approach

Our goal is to systematically compare different sentiment analysis models by evaluating their trade-offs between performance and computational efficiency (training/inference time). We will implement and benchmark the following models:

- Bag-of-Words (Baseline)
- Elastic Net
- Random Forest
- Recurrent Neural Network
- BERT

## Baseline

The baseline will be a traditional, simple model constructed using Bag-of-Words.

## Data

We will use a dataset from [Hugging Face](https://huggingface.co/datasets/gxb912/large-twitter-tweets-sentiment)

The dataset has a training set with 180’000 rows and a test set with 45’000 rows. Each entry consists of a tweet and a sentiment label (1 = positive, 0 = negative). The tweets are in English. The dataset is slightly imbalanced, with approximately 42% negative and 57% positive sentiments.

## Evaluation Metric

Since this is a binary classification problem, we will use binary cross-entropy as the loss function. Given the relatively balanced distribution (42% negative, 57% positive), we will prioritize Precision, Recall and F1-Score over accuracy.

Additionally, we will use a confusion matrix to analyze the distribution of correct and incorrect predictions. This will allow us to make more detailed statements about model performance.

## Expected Outcome

We aim to produce models that classify the sentiment of a given tweet as either positive or negative. This can be used to gather tweets about a subject and analyze how the Twitter public feels about it.

## References

(To be added during the project)

## Resources

We plan to use a GPU.
