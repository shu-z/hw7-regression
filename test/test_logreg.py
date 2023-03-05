"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np 
from regression import LogisticRegressor, utils
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss



#make some sample data that approximately follows sigmoid 
#to be used by test_prediction and test_training
#this adapted from umair's example for linear regression 

num_points = 1000
w = [2, 3, 0.5]
X = np.random.rand(num_points, len(w) - 1)

noise= np.random.rand(num_points, 1) * 0.01
y_linear = (np.expand_dims(X.dot(w[:-1]), 1) - w[-1] + noise).flatten()

y=(1/(1+np.exp(-y_linear)))
#rescale y between 0 and 1
y = (y-np.min(y))/(np.max(y)-np.min(y))


#first split out train 
train_split = int(0.6 * num_points)
X_train, X_remainder = X[:train_split], X[train_split:]
y_train, y_remainder = y[:train_split], y[train_split:]

#get idx for splitting validation and test 
valid_split= int(len(y_remainder)*0.5)
X_val, X_test = X[train_split:train_split+ valid_split], X[train_split+valid_split:]
y_val, y_test = y[train_split:train_split+ valid_split], y[train_split+valid_split:]

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
X_test = sc.transform(X_test)

#make logreg class 
log_model = LogisticRegressor(num_feats = len(w) - 1, learning_rate=0.01, max_iter=5000, batch_size=50)




def test_prediction():

	log_model.train_model(X_train, y_train, X_val, y_val)
	
	model_pred = log_model.make_prediction(X_test)
	model_pred_binary = np.where(model_pred > 0.5, 1, 0)
	y_test_binary=np.where(y_test > 0.5, 1, 0)


	#get accuracy of prediction on separate test set 
	accuracy=np.sum(model_pred_binary == y_test_binary)/len(y_test_binary)

	#check accuracy is at least a little better than chance lol
	assert accuracy>0.6


	

def test_loss_function():

	#make random y_true and y_pred
	y_true=[0, 1, 1, 0, 1, 1, 0]
	y_pred=[0.1, 0.9, 0.7, 0.2, 0.5, 0.6, 0.3]

	#run loss from our model and sklearn function 
	logreg_loss=log_model.loss_function(np.array(y_true), np.array(y_pred))
	sklearn_loss=log_loss(y_true, y_pred)

	#check that bce loss from our function is close to sklearn loss with reasonable tolerance 
	assert np.isclose(logreg_loss, sklearn_loss, rtol=1e-4), 'BCE loss is not as expected'

def test_gradient():

	#make some sample data 
	X=np.array([[1,1], [2, 5], [5,10]])
	y=np.array([0.25, 0.5, 0.75])

	logmod_testgrad = LogisticRegressor(num_feats=3)
	#need to add W for make_prediction
	logmod_testgrad.W = np.array([0,0])

	grad = logmod_testgrad.calculate_gradient(y, X)

	#compare to manually calcualted values
	assert np.allclose (grad, np.array([-1/3, -3/4])), 'Gradient value is not as expected'


	

def test_training():
	pass







	#check loss history


	#check weights are being updated 



	#log_model.loss_history_val[0]