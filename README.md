**Motion Prediction**

This is one of my masters project that deals with predicting the motion of vehicle using a dataset of several recordings from different junctions.

Accomplished:
Completed the data processing of raw data set before training the model. The steps in data processing include:-
 a) Down sizing - Reducing data to half.
 b) Label encoding - Converting class labels to numerical data. 
 c) Normalizing - Normalizing data to a range between 0 and 1.
The modified dataset is then stored in pickle format, so that it can be loaded again without repeating the same step of data processing if the recording id inputted is same.
Callbacks are used to store the best 2 training and validation loss in a training.


Pending work:
Training and testing using Constant velocity model.

Target:
Implementing motion prediction using 4 or 5 models. The models include Multi layer perceptron, constant velocity model and so on.
