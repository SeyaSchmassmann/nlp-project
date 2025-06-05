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
TODO Text

#### Implementation
TODO Text

#### Results
TODO Text

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