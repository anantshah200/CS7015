# Program : Backpropagation with the help of numpy and pandas
# Date : 31-1-2019
# Author ; Anant Shah
# E-Mail : anantshah200@gmail.com

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

NUM_FEATURES = 784
NUM_CLASSES = 10
NUM_COMPONENTS = 784 # Top dimensions for PCA
X_COMPONENT = 28
Y_COMPONENT = 28

def get_specs() :
	"Function to get the specifications from the command line arguments"
	parser = argparse.ArgumentParser()
	parser.add_argument("--lr",type=float,help="initial learning rate of the algorithm")
	parser.add_argument("--momentum",type=float,help="momentum to be used by any momentum based algorithm")
	parser.add_argument("--num_hidden",type=int,help="number of hidden layers in the neural network")
	parser.add_argument("--sizes",nargs="?")
	parser.add_argument("--activation",help="type of activation function in the layers")
	parser.add_argument("--loss",help="type of loss at the output layer")
	parser.add_argument("--opt",help="type of optimization algorithm")
	parser.add_argument("--batch_size",type=int,help="size of each mini-batch")
	parser.add_argument("--epochs",type=int,help="number of time steps for which we train our model")
	parser.add_argument("--anneal",help="set to either true or false")
	parser.add_argument("--save_dir",help="the directory where pickled model should be saved")
	parser.add_argument("--expt_dir",help="the directory in which the log files will be saved")
	parser.add_argument("--train",help="path to the training dataset")
	parser.add_argument("--val",help="path to the validation dataset")
	parser.add_argument("--test",help="path to the test dataset")

	args = parser.parse_args()
	return args

def save_weights(theta, epoch, save_dir) :
	with open(save_dir +'weights_{}.pkl'.format(epoch),'wb') as f:
		pickle.dump(theta,f)

def load_weights(state, save_dir) :
	with open(save_dir +'weights_{}.pkl'.format(state),'rb') as f :
		theta = pickle.load(f)
	return theta

def initialize_parameters(num_hidden,sizes) :
	# Function to randomly initialize the parameters 
	# Parameters : num_hidden - Number of hidden layers  
	#	       sizes - a list with number of perceptrons in each hidden layer including the input layer and output layer

	np.random.seed(1234)
	theta = {}
	for i in range(1,num_hidden+2) :
		theta["W"+str(i)] = np.random.randn(sizes[i],sizes[i-1]) * np.sqrt(2./(sizes[i] + sizes[i-1])) # Xavier Initialization
		theta["b"+str(i)] = np.zeros((sizes[i],1),dtype=np.float64)
	
	return theta

def initialize_updates(sizes,update) :
	# Function to initialize the update matrices to 0
	# Parameters -  updates - A dictionary which contains the history of updates for each parameter
	#		sizes - The size of each layer in the network
	num_layers = sizes.shape[0]
	for i in range(1,num_layers) :
		update["W"+str(i)] = np.zeros((sizes[i],sizes[i-1]),dtype=np.float64) 
		update["b"+str(i)] = np.zeros((sizes[i],1),dtype=np.float64)
	return update

def create_mini_batch(N, batch_size) :
	# Function to obtain randomized batch indices for the mini-batch optimization algorithm
	# Parameters -  N - number of training examples
	#		batch_size - The size of a batch(has to be a multiple of 5)

	indices = np.random.permutation(N)
	mini_batch_indices = [] # A list containing each set of indices
	for i in range(int(N/batch_size)) :
		mini_batch_indices.append(indices[i*batch_size:(i+1)*batch_size])
	return mini_batch_indices

def sigmoid(A) :
	# Implement the logistic function
	return 1.0/(1.0 + np.exp(-A))

def tanh(A) :
	# Implement the tanh function
	return np.tanh(A)

def relu(A) :
	# Implements the non-linear relu function
	index = np.where(A<0)
	A[index] = A[index]*0.01
	return A

def softmax(A) :
	# Given a matrix of values, this function will return the softmax values
	m = A.shape[0]
	n = A.shape[1]
	A_exp = np.exp(A) # Found the exponential of all the values
	e_sum = np.sum(A_exp,axis=0,keepdims=True)
	A_softmax = A_exp/e_sum
	assert A_softmax.shape == (m,n)
	return A_softmax

def forward_layer(H_prev,activation,W,b,layer,cache) :
	# Function to perform the forward pass of a hidden layer in the neural network
	# Parameters :  A_prev - The activation obtained from the previous layer(acting as X for the next layer)
	#	        activation - THe type of activation function in the layer
	#		W - Weight matrix of the current layer
	#		b - bias vector of the current layer
	#		layer - The layer in the neural network(starting from 1)
	#		cache - Dictionary to store the activation values required for the backward pass

	N = H_prev.shape[1] # Number of training examples
	assert H_prev.shape[0] == W.shape[1]
	assert b.shape == (W.shape[0],1)

	A = np.dot(W,H_prev)+b
	cache["A"+str(layer)] = A
	if(activation == "sigmoid") :
		H = sigmoid(A)
		cache["H"+str(layer)] = H
	elif (activation == "tanh") :
		H = tanh(A)
		cache["H"+str(layer)] = H
	elif (activation == "relu") :
		H = relu(A)
		cache["H"+str(layer)] = H

	assert H.shape == (W.shape[0],N)
	return H, cache

def feed_forward(X,activation,theta,sizes) :
	# Function to implement the forward pass throughout the whole network
	# Parameters :  X - Input Data
	#		activation - The type of activation in the hidden layers
	#		parameters - Dictionary which contains the parameters of each fully connected layer
	#		sizes - array containing the size of each layer in the network
	#		cache - Dictionary to store the activation values required for the backward pass

	N = X.shape[1] # Number of training examples
	layers = sizes.shape[0] # Total number of layers = hidden_layers + input_layer + output_layer
	H_prev = X
	cache = {}
	cache["H0"] = X

	for i in range(1,layers-1) :
		W = theta["W"+str(i)]
		b = theta["b"+str(i)]
		H, cache = forward_layer(H_prev,activation,W,b,int(i),cache)
		H_prev = H
	
	# The last layer has a softmax function hence we need to perform the computation separately
	b = theta["b"+str(layers-1)]
	W = theta["W"+str(layers-1)]
	A = np.dot(W,H_prev)+b
	cache["A"+str(layers-1)] = A
	Y_hat = softmax(A)
	assert Y_hat.shape == (sizes[layers-1],N)
	return Y_hat, cache

def cost(Y,Y_hat,loss,theta,reg) :
	# Function to calculate the error in prediction by the neural network
	# Parameters :  Y - Actual labels for the data
	#		Y_hat - Predicted probabilities of the labels
	#		loss - The type of loss (either square error or entropy)
	#		theta - dictionary containing the parameter vectors
	#		reg - the regularization parameter
	assert Y.shape == Y_hat.shape
	N = Y.shape[1]
	if (loss == "sq") :
		error = np.sum((Y-Y_hat)**2)/(2*N)
	elif (loss == "ce") :
		error = -np.sum(np.multiply(Y,np.log(Y_hat))) / N
	L = int(len(theta)/2)
	reg_error = 0.0
	for i in range(1,L+1) :
		reg_error = reg_error + (reg/(2*N))*np.sum(np.square(theta["W"+str(i)])) # L2 norm regularization
	error = error + reg_error
	error = np.squeeze(error)
	return error

def back_layer(layer,cache,grads,theta,activation,reg) :
	# Function to compute the gradient of the loss with respect to the pre-activation of a layer
	# Parameters : layer - The current layer we are acting on
	#		cache - The data stored from the forward propagation step 
	#		grads - A dictionary containing the gradients of the loss w.r.t the parameters in the network
	#		theta - A dictionary containing the weights and biases
	#		activation - The type of activation in a layer of the neural network
	# 		reg - The regularization factor for the weights

	dH = grads["dH"+str(layer)] # Gradient of the loss with respect to the activations of a <layer>
	N = dH.shape[1]
	A = cache["A"+str(layer)]
	H_prev = cache["H"+str(layer-1)]
	W = theta["W"+str(layer)]
	# We will be using the chain rule. It will lead to a point-wise multiplication as there is only 1 path from the pre-activation to the post-activation of a layer
	if activation=="tanh" :
		dA = np.multiply(dH,(1-tanh(A)**2))
	elif activation=="sigmoid" :
		dA = np.multiply(dH,np.multiply(sigmoid(A),(1-sigmoid(A))))
	elif activation=="relu" :
		leak = 0.01
		index_p = np.where(A>=0)
		index_n = np.where(A<0)
		dHdA = np.zeros(dH.shape)
		dHdA[index_p] = 1
		dHdA[index_n] = leak
		dA = np.multiply(dH, dHdA)

	dW =(1./N) * np.dot(dA,H_prev.T) + (reg/N)*W
	db =(1./N) * np.sum(dA,axis=1,keepdims=True)
	dH_prev = np.dot(W.T,dA)
	grads["dA"+str(layer)] = dA
	grads["dW"+str(layer)] = dW
	grads["db"+str(layer)] = db
	grads["dH"+str(layer-1)] = dH_prev
	return grads

def back_prop(X,Y,Y_hat,loss,cache,theta,activation,sizes,reg) :
	# Function to backpropagate through the network to update the weights
	# Parameters -  X - The input data
	#		Y - The output data
	#		Y_hat - Probability distribution of the predicted class
	#		error - The loss obtained after a forward propagation step
	#		loss - The type of loss(cross entropy or squared error)
	#		cache - The data stored from the forward propagation step
	#		grads - A dictionary containing the gradients of the loss w.r.t the parameters in the network
	#		theta - A dictionary containing the weights and biases
	#		activation - The type of activation in a layer of the neural network
	#		sizes - A vector containing the number of neurons in each layer
	#		reg - The regularization parameter

	# First, we need to calculate the derivative of the loss function w.r.t the output layer
	
	layers = int(sizes.shape[0])
	N = X.shape[1]
	grads = {}

	if loss=="sq" :
		grads["dH"+str(layers-1)] = (Y_hat-Y) # The loss is the avreage loss and hence we include the <m> variable
		true_class = np.where(Y.T==1)
		Y_L = Y_hat[true_class[1],true_class[0]]
		Y_L = np.tile(Y_L,(NUM_CLASSES,1)) 
		Y_sqsum = np.tile(np.sum(Y_hat**2,axis=0,keepdims=True),(NUM_CLASSES,1))
		grads["dA"+str(layers-1)] = (Y_hat) * (-Y_sqsum+Y_hat-Y+Y_L)
	elif loss=="ce" :
		grads["dH"+str(layers-1)] = -(Y/Y_hat)
		grads["dA"+str(layers-1)] = -(Y-Y_hat)
	
	H_prev = cache["H"+str(layers-2)]
	dA = grads["dA"+str(layers-1)]
	grads["dW"+str(layers-1)] = (1./N) * np.dot(dA,H_prev.T) + (reg/N)*theta["W"+str(layers-1)]
	grads["db"+str(layers-1)] = (1./N) * np.sum(dA,axis=1,keepdims=True)
	grads["dH"+str(layers-2)] = np.dot(theta["W"+str(layers-1)].T,dA)
	# We have now obtained the gradients at the output layer. We just need to backpropagate through the network to find the gradients of the loss w.r.t the parameters

	for i in range(layers-2,0,-1) :
		grads = back_layer(int(i),cache,grads,theta,activation,reg)
	
	# All of the gradients of the loss w.r.t the parameters have been added to the grads dictionary
	# Need to update the parameters after this
	return grads

def optimize(theta,grads,update,mom,update_t,mom_t,time_step,learning_rate,momentum,algo) :
	# Function to perform a certain optimization algorithm with the dictionary of gradients given
	# Parameters - theta - The dictionary of weights and biases which need to be updated
	#		grads - The dictionary of gradients which will be used to update the parameters 
	#		update - A dictionary containing the update values at instant <t-1> for the momentum based algorithm
	#		mom - A dictionary containing the history of the sum of gradients for the adam algorihm
	#		time_step - The current epoch at which the optimization is taking place
	#		learning_rate - Rate to be used by the gradient descent algorithm; gradients will be scaled by this factor
	#		momentum - The momentum parameter for a momentum based algorithm
	#		algo - Type of optimization to be used(gradient descent, rmsprop, etc.)
	L = int(len(theta)/2) # Number of layers in the network excluding the input layer
	if algo == "gd" :
		for i in range(1,L+1) :
			theta["W"+str(i)] = theta["W"+str(i)] - learning_rate*grads["dW"+str(i)]
			theta["b"+str(i)] = theta["b"+str(i)] - learning_rate*grads["db"+str(i)]
	elif algo == "momentum" :
		for i in range(1,L+1) :
			update["W"+str(i)] = momentum*update["W"+str(i)] + learning_rate*grads["dW"+str(i)]
			update["b"+str(i)] = momentum*update["b"+str(i)] + learning_rate*grads["db"+str(i)]
			theta["W"+str(i)] = theta["W"+str(i)] - update["W"+str(i)]
			theta["b"+str(i)] = theta["b"+str(i)] - update["b"+str(i)]
	elif algo == "nag" :
		for i in range(1,L+1) :
			update["W"+str(i)] = momentum*update["W"+str(i)] + learning_rate*grads["dW"+str(i)] # grads will be calculated differently for this case
			update["b"+str(i)] = momentum*update["b"+str(i)] + learning_rate*grads["db"+str(i)]
			theta["W"+str(i)] = theta["W"+str(i)] - update["W"+str(i)]
			theta["b"+str(i)] = theta["b"+str(i)] - update["W"+str(i)]
	elif algo == "adam" :
		beta1 = 0.9
		beta2 = 0.999
		epsilon = 1e-8
		for i in range(1,L+1) :
			mom["W"+str(i)] = beta1*mom["W"+str(i)] + (1-beta1)*grads["dW"+str(i)]
			mom_t["W"+str(i)] = mom["W"+str(i)] / (1 - beta1**time_step)
			update["W"+str(i)] = beta2*update["W"+str(i)] + (1-beta2)*(grads["dW"+str(i)]**2)
			update_t["W"+str(i)] = update["W"+str(i)] / (1 - beta2**time_step)
			mom["b"+str(i)] = beta1*mom["b"+str(i)] + (1-beta1)*grads["db"+str(i)]
			mom_t["b"+str(i)] = mom["b"+str(i)] / (1 - beta1**time_step)
			update["b"+str(i)] = beta2*update["b"+str(i)] + (1-beta2)*(grads["db"+str(i)]**2)
			update_t["b"+str(i)] = update["b"+str(i)] / (1 - beta2**time_step)
			theta["W"+str(i)] = theta["W"+str(i)] - learning_rate*mom_t["W"+str(i)] / (np.sqrt(update_t["W"+str(i)])+epsilon)
			theta["b"+str(i)] = theta["b"+str(i)] - learning_rate*mom_t["b"+str(i)] / (np.sqrt(update_t["b"+str(i)])+epsilon)
	return theta, update, mom, update_t, mom_t

def train(X, Y, X_val, Y_val, sizes, learning_rate, momentum, activation, loss, algo, batch_size, epochs, anneal, reg, save_dir) :
	# Function to train the model to identify classes in the dataset
	# Parameters -  X - input data
	#		Y - the actual classes
	#		X_val - the validation dataset
	#		Y_val - the validation dataset
	#		sizes - A list consisting of the sizes of each layer in the network
	#		learning_rate - The learning-rate for the gradient descent optimization algorithm
	#		momentum - The momentum used for a momentum based optimization algorithm
	# 		activstion - The type of activation in the neural network
	#		loss - The type of loss function used
	#		algo - The optimization algorithm used
	#		batch_size - Batch size for batch optimization
	#		epochs - NUmber of iterations for which we train the model
	#		anneal - Anneal if true
	#		reg - The regularization parameter

	N = X.shape[1] # Number of training examples
	num_hidden = sizes.shape[0] - 2

	if (batch_size%5!=0 and batch_size!=1) :
		print("error : Invalid batch size - should be a multiple of 5 or 1")
		sys.exit()

	num_batches = N/batch_size

	# First initialize the parameters
	theta = initialize_parameters(num_hidden,sizes)
	update = {}
	mom = {}
	update_t = {}
	mom_t = {}
	update = initialize_updates(sizes,update) # Initialize all the updates to 0
	update_t = initialize_updates(sizes,update_t)
	mom = initialize_updates(sizes,mom)
	mom_t = initialize_updates(sizes,mom_t)
	costs = [] # A list containing the cost after each step
	val_costs = [] # A list containing the validation error after each epoch
	time = 0
	params = {} # A dictionary to hold the copies of all the parameters

	i = 0 # A counter for the indexing in the list storing the validation error
	ep = 0 # Counter for the number of epochs
	patience = 4 # The number of epochs after which we check if the validation loss has decreased or not
	val_count = 0 # Number of epochs for which the validation error has not decreased

	while ep < epochs :
		step = 0
		batch_indices = create_mini_batch(N,batch_size)
		if ep >= 1 :
			params = {"W" : theta.copy(), "G" : grads.copy(), "M" : update.copy(), "V" : mom.copy()}
		for indices in batch_indices :
			X_batch = X[:,indices]
			Y_batch = Y[:,indices]
			Y_hat, cache = feed_forward(X_batch,activation,theta,sizes)
			error = cost(Y_batch,Y_hat,loss,theta,reg)
			if step % 100 == 0 :
				print("Step : " + str(step) + " Epoch : " + str(ep) + " Error : " + str(error) + " Learning Rate : " + str(learning_rate))
			grads = back_prop(X_batch,Y_batch,Y_hat,loss,cache,theta,activation,sizes,reg)
			theta, update, mom, update_t, mom_t = optimize(theta,grads,update,mom,update_t,mom_t,time+1,learning_rate,momentum,algo)
			time = time + 1
			step = step + 1

		Y_val_hat, trash = feed_forward(X_val,activation,theta,sizes) # Calculating the error on the validation set
		error_val = cost(Y_val,Y_val_hat,loss,theta,reg) # Calculating the error on the validation set

		if (anneal == "true") and (i>=1) :
			if error_val > val_costs[i-1] :
				# If the validation error increases, re-run the whole epoch
				val_count = val_count + 1
				learning_rate = learning_rate / 2
				theta = params["W"]
				grads = params["G"]
				update = params["M"]
				mom = params["V"]
				time = time - int(N/batch_size)
			else :
				i = i + 1
				val_count = 0
				val_costs.append(error_val)
			save_weights(theta,ep,save_dir)
		elif (anneal == "false") and (i>=1) :
			if error_val > val_costs[i-1] :
				val_count = val_count + 1
			else :
				val_count = 0
				i = i + 1
				val_costs.append(error_val)
			save_weights(theta,ep,save_dir)

		if i==0 :
			save_weights(theta,ep,save_dir)
			i = i + 1
			val_costs.append(error_val)
		
		if (val_count == patience) :
			theta = load_weights(ep-patience,save_dir)
			return theta

		costs.append(error)
		ep = ep + 1

	print("Training complete")
	return theta

def test_accuracy(X_test,Y_test,theta,activation,sizes) :
	# Function to test the accuracy of the trained model
	Y_hat, trash = feed_forward(X_test,activation,theta,sizes)
	N = X_test.shape[1]
	max_val = 0

	for i in range(N) :
		max_ind = np.where(Y_hat[:,i] == np.amax(Y_hat[:,i]))
		min_ind = np.where(Y_hat[:,i] != np.amax(Y_hat[:,i]))
		if(len(max_ind[0]) >= 2) :
			max_val = max_ind[0][0]
			min_ind = np.append(min_ind[0],max_ind[0][1:])
		else :
			max_val = max_ind[0]
		Y_hat[max_val,i] = 1
		Y_hat[min_ind,i] = 0
	accuracy = accuracy_score(Y_test.T,Y_hat.T)
	return accuracy

def test_model(X_test,theta,activation,sizes) :
	# Function to predict the outcomes of the test data
	Y_hat, trash = feed_forward(X_test,activation,theta,sizes)
	N = X_test.shape[1]
	max_val = 0
	predictions = np.zeros((N,1)).astype(int)
	
	for i in range(N) :
		max_ind = np.where(Y_hat[:,i] == np.amax(Y_hat[:,i]))
		if (len(max_ind[0]) >= 2) :
			max_val = max_ind[0][0]
		else :
			max_val = max_ind[0]
		predictions[i] = max_val
	index = (np.arange(0,N).T).reshape(N,1)
	predictions = np.hstack((index,predictions))
	return predictions

def get_data(train_path,val_path,test_path) :
	# Function to get the data for training
	# Augment data so as to obtain more training examples by left shifting the pixels by 1
	train_data = pd.read_csv(train_path)
	val_data = pd.read_csv(val_path)
	test_data = pd.read_csv(test_path)
	train = np.array(train_data)
	val = np.array(val_data)
	test = np.array(test_data)

	N = train.shape[0]
	X_train = train[:,1:NUM_FEATURES+1].T / 255.0
	eps = 1e-8
	#X_train = (X_train - np.mean(X_train))/np.sqrt(np.var(X_train)+eps)

	#Augment the data by shifting the picture upwards by 1 pixel
	X_aug_temp = np.reshape(X_train[:,0:int(N/4)].T,(int(N/4),X_COMPONENT,Y_COMPONENT),order='C')
	X_aug_temp = np.roll(X_aug_temp[:],1,axis=1)
	X_aug = np.reshape(X_aug_temp,(int(N/4),NUM_FEATURES)).T
	X_bshift_temp = np.reshape(X_train[:,int(N/4):int(N/2)].T,(int(N/4),X_COMPONENT,Y_COMPONENT),order='C')
	X_bshift_temp = np.roll(X_bshift_temp[:],-1,axis=1)
	X_bshift = np.reshape(X_bshift_temp,(int(N/4),NUM_FEATURES)).T
	X_rshift_temp = np.reshape(X_train[:,int(N/2):int(3*N/4)].T,(int(N/4),X_COMPONENT,Y_COMPONENT),order='C')
	X_rshift_temp = np.roll(X_rshift_temp[:],1,axis=2)
	X_rshift = np.reshape(X_rshift_temp,(int(N/4),NUM_FEATURES)).T
	X_lshift_temp = np.reshape(X_train[:,int(3*N/4):N].T,(int(N/4),X_COMPONENT,Y_COMPONENT),order='C')
	X_lshift_temp = np.roll(X_lshift_temp[:],-1,axis=2)
	X_lshift = np.reshape(X_lshift_temp,(int(N/4),NUM_FEATURES)).T
	assert X_aug.shape == (NUM_FEATURES,int(N/4))
	assert X_bshift.shape == (NUM_FEATURES,int(N/4))
	assert X_rshift.shape == (NUM_FEATURES,int(N/4))
	assert X_lshift.shape == (NUM_FEATURES,int(N/4))
	assert X_bshift.shape == X_aug.shape

	X_train = np.hstack((X_train,X_aug))
	X_train = np.hstack((X_train,X_bshift))
	X_train = np.hstack((X_train,X_rshift))
	X_train = np.hstack((X_train,X_lshift))
	assert X_train.shape == (NUM_FEATURES,2*N)

	Y_train = train[:,NUM_FEATURES+1,None].T
	Y_st = np.zeros((NUM_CLASSES-1,N)).astype(int)
	Y_train = np.vstack((Y_train,Y_st))
	init_index = np.where(Y_train[0,:] == 0)
	for i in range(1,NUM_CLASSES) :
		Y_train[i,np.where(Y_train[0][:] == int(i))] = 1
	Y_train[0,np.where(Y_train[0][:] != 0)] = 0
	Y_train[0,init_index] = 1
	Y_train = np.hstack((Y_train,Y_train))
	assert Y_train.shape == (NUM_CLASSES,2*N)

	X_val = val[:,1:NUM_FEATURES+1].T / 255.0
	#X_val = (X_val - np.mean(X_val))/np.sqrt(np.var(X_val))
	assert X_val.shape == (NUM_COMPONENTS,val.shape[0])

	Y_val = val[:,NUM_FEATURES+1,None].T
	Y_st = np.zeros((NUM_CLASSES-1,val.shape[0])).astype(int)
	Y_val = np.vstack((Y_val,Y_st))
	init_val = np.where(Y_val[0,:] == 0)
	for i in range(1,NUM_CLASSES) :
		Y_val[i,np.where(Y_val[0][:] == int(i))] = 1
	Y_val[0,np.where(Y_val[0][:] != 0)] = 0
	Y_val[0,init_val] = 1
	assert Y_val.shape == (NUM_CLASSES,val.shape[0])

	X_test = test[:,1:NUM_FEATURES+1].T / 255.0
	#X_test = (X_test - np.mean(X_test))/np.sqrt(np.var(X_test))
	assert X_test.shape == (NUM_COMPONENTS,test.shape[0])

	data = {"X_train" : X_train,"Y_train" : Y_train,"X_val" : X_val,"Y_val" : Y_val,"X_test" : X_test}

	return data

# Assign variables to the parameters from the command line input
args = get_specs()
learning_rate = args.lr
momentum = args.momentum
reg = 0.002
num_hidden = args.num_hidden
hidden_sizes = args.sizes

sizes = []
sizes.append(NUM_COMPONENTS)
for num in hidden_sizes.split(',') :
	sizes.append(int(num))
sizes.append(NUM_CLASSES)
sizes = np.array(sizes)

activation = args.activation
loss = args.loss
algo = args.opt
batch_size = args.batch_size
epochs = args.epochs
anneal = args.anneal
train_path = args.train
val_path = args.val
test_path = args.test
save_dir = args.save_dir

data = get_data(train_path, val_path, test_path)
X_train = data["X_train"]
Y_train = data["Y_train"]
X_val = data["X_val"]
Y_val = data["Y_val"]
X_test = data["X_test"]

theta = train(X_train,Y_train,X_val,Y_val,sizes, learning_rate, momentum, activation, loss, algo, batch_size, epochs, anneal, reg, save_dir)
accuracy = test_accuracy(X_val,Y_val,theta,activation,sizes)
train_accuracy = test_accuracy(X_train,Y_train,theta,activation,sizes)
test_out = test_model(X_test,theta,activation,sizes)
pd.DataFrame(test_out).to_csv("submission.csv", header=["id","label"],index=False)
print("Accuracy :"+str(accuracy))
print("Training Accuracy :"+str(train_accuracy))
