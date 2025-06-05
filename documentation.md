# Project documentation
This document provides the documentation for the Sentiment Analysis of Tweets project during the NLP course at FHNW. The different approaches and models used in this project are described in theoretical terms and the results are presented in a structured manner.

## Data Analysis
In this project the "Large-Twitter-Tweets-Sentiment" Dataset has been used, which is available on [Hugging Face](https://huggingface.co/datasets/gxb912/large-twitter-tweets-sentiment).

The dataset consists of tweets that have been labeled positive or negative. The dataset is divided into a training set and a test set, the training set contains 179'995 rows, the test set contains 44'999 rows. Each row contains a tweet and its corresponding label (1 = positive, 0 = negative).

In both the training and test set, the positive sentiment has a higher representation than the negative sentiment. The training set contains 57.8488% positive and 42.1512% negative tweets, while the test set contains 57.8502% positive and 42.1498% negative tweets.

<img src="resources/sentiment_distribution_train.png" />
<img src="resources/sentiment_distribution_test.png" />

> The source code for the data analysis can be found in the [dataAnalysis.ipynb](data/dataAnalysis.ipynb) notebook.

## Metrics
To evaluate the model-performance, the following metrics are used:

* **Precision**: The ratio of true positive predictions to the total number of positive predictions.
    $$
    \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
    $$

* **Recall**: The ratio of true positive predictions to the total number of actual positive instances.
    $$
    \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
    $$

* **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two metrics.
    $$
    \text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
    $$

* **Training Time**: The time taken to train the model on the training dataset.

* **Inference Time**: The time taken to make predictions on the test dataset.

## Models

### Bag-of-Words
#### Theoretical Background
Bag-of-Words (BoW) is a simple text representation which relies on the frequency of words in a document. In this model, each document is represented as a vector of word counts, ignoring the order of words (sparse vector representation). To use a BoW representation for sentiment analysis, the vecorized words are fed into a machine learning model, such as logistic regression or Naive Bayes, to classify the sentiment of the document.

#### Implementation
The Bag-of-Words (BoW) implementation in this project uses two different vectorization techniques:
* **Count Vectorization**: This technique counts the number of occurrences of each word in the document and creates a sparse matrix representation.

* **TF-IDF Vectorization**: This technique computes the term frequency-inverse document frequency (TF-IDF) score for each word in the document, assigning higher weights to terms that are more informative for a specific document relative to the entire corpus.

The model is then trained using various machine learning algorithms:
* **Naive Bayes**: A probabilistic classifier based on Bayes' theorem. It assumes that the features $x_1, x_2, ..., x_n$ are conditionally independent given the class label $y$. The predicted class is the one that maximizes the posterior probability:
  $$
  \hat{y} = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i \mid y)
  $$

* **Logistic Regression**: A linear model that uses the sigmoid   (logistic function) to model the probability of a binary outcome. The probability of a tweet being positive is calculated  as:
  $$
  P(y=1 \mid x) = \frac{1}{1 + e^{-(w^T x + b)}}
  $$
    where $ w $ is the weight vector, $ x $ is the feature vector, and $ b $ is the bias term.

* **Linear Support Vector Machine (SVM)**: A linear classifier that identifies the hyperplane which maximizes the margin between classes in the feature space. The model is trained using the hinge loss function:
  $$
  L(y, f(x)) = \max(0, 1 - y \cdot f(x))
  $$
  where $ y $ is the true label, $ f(x) $ is the predicted score, and $ x $ is the feature vector.

* **XGBoost**: An optimized gradient boosting algorithm that uses decision trees as base learners. Each new tree is trained to minimize a regularized objective function, improving the model’s predictive accuracy while preventing overfitting. The objective at iteration $ t $ is:
  $$
  \mathcal{L}^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)
  $$
  where $ l $ is a loss function (e.g., logistic loss), $ f_t $ is the new decision tree, and $ \Omega $ is a regularization term penalizing model complexity.

#### Results
The in the previous section described models were trained on the training set and evaluated on the test set. Each model was tested with both vectorization techniques (Count Vectorization and TF-IDF Vectorization).

##### Precision
The highest test precision was achived using a count vectorizer and a Multinomial Naive Bayes model. The precision of the different models ranged from 0.720 and 0.788.

<img src="resources/bow_test_precision.png" height="300" />

##### Recall
The highest test recall was achived with a TF-IDF vectorizer and XGBoost model. 
The recall of the different models ranged from 0.802 and 0.887.

<img src="resources/bow_test_recall.png" height="300" />

##### F1-Score
The highest f1-score was achived with a TF-IDF vectorizer and Logistic Regression model. The f1-score of the different models ranged from 0.795 and 0.808.

<img src="resources/bow_test_f1.png" height="300" />

##### Training Time
TODO Text

##### Inference Time
TODO Text

##### Summary
Among all tested configurations:
* Multinomial Naive Bayes with Count Vectorization achived the highest precision.
* XGBoost with TF-IDF Vectorization achived the highest recall.
* Logistic Regression with TF-IDF Vectorization achived the highest f1-score.

Considering the trade-off between precision, recall and f1-score, the Logistic Regression model with TF-IDF Vectorization can be considered the most effective overall for this Bag-of-Words based sentiment classification task.

> The source code for the Bag-of-Words model can be found in the [bagOfWords.ipynb](models/bagOfWords.ipynb) notebook.

### Elastic Net
#### Theoretical Background
TODO Text

#### Implementation
TODO Text

#### Results
TODO Text

> The source code for the Elastic Net model can be found in the [elasticNet.ipynb](models/elasticNet.ipynb) notebook.

### Random Forest
#### Theoretical Background
TODO Text

#### Implementation
TODO Text

#### Results
TODO Text

> The source code for the Random Forest model can be found in the [randomForest.ipynb](models/randomForest.ipynb) notebook.

### Recurrent Neural Network
#### Theoretical Background
TODO Text

#### Implementation
TODO Text

#### Results
TODO Text

> The source code for the Recurrent Neural Network model can be found in the [rnn.ipynb](models/rnn.ipynb) notebook.

### LSTM
#### Theoretical Background
TODO Text

#### Implementation
TODO Text

#### Results
TODO Text

> The source code for the LSTM model can be found in the [lstm.ipynb](models/lstm.ipynb) notebook.

### BERT
#### Theoretical Background
TODO Text

#### Implementation
TODO Text

#### Results
TODO Text

> The source code for the BERT model can be found in the [bert.ipynb](models/bert.ipynb) notebook.

## Discussion