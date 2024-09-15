**Motion Prediction**

This is one of my masters project that deals with predicting the motion of vehicle using a dataset of several recordings from different junctions.

Accomplished:
Completed the data processing of raw data set before training the model. The steps in data processing include:-
 a) Down sizing - Reducing data to half.
 b) Label encoding - Converting class labels to numerical data. 
 c) Normalizing - Normalizing data to a range between 0 and 1.
The modified dataset is then stored in pickle format, so that it can be loaded again without repeating the same step of data processing if the recording id inputted is same.
Callbacks are used to store the best 2 training and validation loss in a training.
Added the algorithm for 6 models:- 1) MLP 2) LSTM 3) GRU (NN based models) 4) Constant Velocity 5) Constant Acceleration 6) Kinematic Bicycle Model(Physics based models).
Implementation completed for training the model and testing the trained model with different dataset.

Pending work:
Complete the plotting of test loss of models

