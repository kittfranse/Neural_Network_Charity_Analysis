# Neural Network Charity Analysis
***Neural Networks and Deep Learning***

## Overview

The purpose of this analysis is to process metadata associated with the charitable arm of this client. The model will take in a variety of data about the organizations and their donation history and figure out which organizations with what kind of General profile are likely to be the best organizations to donate to.  We created deep neural network to analyze the metadata and understand the success rate of past donations.

## Results
 I was with unsupervised machine learning, All of the data points have to be numerical. To mitigate this, all of the name columns were dropped and then the categorical data that was in string format was coded into binary format and those subsets were appended to the original data frame. Then, the **IS_SUCCESSFUL** column was used as the ex Datapoint and the remainder of the data frame was used as a way that appointment. This is split into test data and training data. And then the model was run on the training data and measured against the test data.

 ```python
 # Split our preprocessed data into our features and target arrays
X = application_df.drop('IS_SUCCESSFUL', axis = 1)
y = application_df.IS_SUCCESSFUL

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, stratify = y)
 ```


##Summary

Unfortunately, my model was not able to be optimized. In fact, as I added more layers to my neural networks, I got less and less accuracy in my overall score.  As expected, the add layers strategy was honestly the best one. I think to better understand this model, visualizing the data might be useful to know which kind of function my provide the best modeling per layer. Interestingly, changing the second layer function from Relu to Tanh was very useful.
