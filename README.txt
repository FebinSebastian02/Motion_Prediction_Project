***Motion Prediction - Group 27***


Things to know: a) Place the dataset folder inside data_processing folder.
		b) venv folder inside main code directory has been removed to reduce size of project for uploading.
**********************************************************************************************************************************************

For Test loss calculation of Physics based models:-
1) In utils.py file, set the data path where the dataset is stored.

2) Choose between the stages:-
	a) The stage should be "test" only.

3) Choose the recording id you need. I have used:-
	a) 18 for all testing.

4) Choose the model type you want to use. The model types are abbreviated as:-
	a) CVM - Constant Velocity model
	b) CAM - Constant Acceleration model
	c) BCM - Bicycle model

5) Finally, run the "main.py" file and the test loss can be observed.

6) Check test_loss_results folder to see the csv files of the predicted values and the actual values

***********************************************************************************************************************************************

For Test loss calculation of Neural network based models:-
1) In utils.py file, set the data path where the dataset is stored.

2) Choose between the stages:-
	a) For Neural network models such as MLP, LSTM, GRU, the stage should be "fit" first.

3) Choose the recording id you need. I have used:-
	a) 21, 22 for all training.

4) Choose the model type you want to use. The model types are abbreviated as:-
	a) MLP - Multi layer perceptron
	b) LSTM - Long short term memory
	c) GRU - Gated recurrent unit

5) You can select the epochs of your choice.

6) Finally, run the "main.py" file and the training and validation loss can be observed.

7) Now, Go to the following directory:- "code\code\training_and_testing\logs\wandb_logs\SS2024_motion_prediction". 
	a) Select the folder that got modified latest. 
	b) From the checkpoints folder inside it, choose the file with file name "model type-checkpoint.ckpt". e.g.:- "GRU-checkpoint.ckpt".
	c) Copy / Replace this file to "code\code\training_and_testing\logs" directory.

8) Change the stage to "test" in the main.py.

9) Choose the recording id you need. I have used:-
	a) 18 for all testing.

10) Finally, run the "main.py" file and the test loss can be observed.
