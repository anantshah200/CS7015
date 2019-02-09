# Program : Backpropagation with the help of numpy and pandas
# Date : 31-1-2019
# Author ; Anant Shah
# E-Mail : anantshah200@gmail.com

import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import argparse

def get_specs() :
	"Function to get the specifications from the command line arguments"
	parser = argparse.ArgumentParser()
	parser.add_argument("--lr",type=float,help="initial learning rate of the algorithm")
	parser.add_argument("--momentum",type=float,help="momentum to be used by any momentum based algorithm")
	parser.add_argument("--num_hidden",type=int,help="number of hidden layers in the neural network")
	parser.add_argument("--sizes",nargs="?")
	parser.add_argument("--activation",help="type of activation function in the layers")


def initialize_parameters(num_hidden,sizes) :
	# Function to randomly initialize the parameters 
	# Parameters : num_hidden - Number of hidden layers  
	#	       sizes - a list with number of perceptrons in each hidden layer including the input layer and output layer

	np.random.seed(1234)
	theta = {}
	for i in range(1,num_hidden+2) :
		theta["W"+str(i)] = np.random.randn(sizes[i],sizes[i-1])
		theta["b"+str(i)] = np.zeros((sizes[i],1))
	
	return theta

def sigmoid(A) :
	# Implement the logistic function
	return 1.0/(1.0 + np.exp(-A))

def tanh(A) :
	# Implement the tanh function
	return np.tanh(A)

def softmax(A) :
	# Given a matrix of values, this function will return the softmax values
	m = A.shape[0]
	n = A.shape[1]
	A_exp = np.exp(A) # Found the exponential of all the values
	e_sum = np.sum(A_exp,axis=0)
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

	m = H_prev.shape[1] # Number of training examples
	assert H_prev.shape[0] == W.shape[1]

	A = np.dot(W,H_prev)+b
	cache["A"+str(layer)] = A
	if(activation == "sigmoid") :
		H = sigmoid(A)
		cache["H"+str(layer)] = H
	elif (activation == "tanh") :
		H = tanh(A)
		cache["H"+str(layer)] = H

	assert H.shape == (W.shape[0],m)
	return H

def feed_forward(X,activation,theta,sizes,cache) :
	# Function to implement the forward pass throughout the whole network
	# Parameters :  X - Input Data
	#		activation - The type of activation in the hidden layers
	#		parameters - Dictionary which contains the parameters of each fully connected layer
	#		sizes - array containing the size of each layer in the network
	#		cache - Dictionary to store the activation values required for the backward pass

	m = X.shape[1] # Number of training examples
	layers = sizes.shape[0] # Total number of layers = hidden_layers + input_layer + output_layer
	H_prev = X

	for i in range(1,layers-1) :
		W = theta["W"+str(i)]
		b = theta["b"+str(i)]
		H = forward_layer(H_prev,activation,W,b,i,cache)
		H_prev = H
	
	# The last layer has a softmax function hence we need to perform the computation separately
	b = theta["b"+str(layers-1)]
	W = theta["W"+str(layers-1)]
	A = np.dot(W,H_prev)+b
	cache["A"+str(layers-1)] = A
	Y_hat = softmax(A)
	assert Y_hat.shape == (sizes[layers-1],m)
	return Y_hat

def cost(Y,Y_hat,loss) :
	# Function to calculate the error in prediction by the neural network
	# Parameters :  Y - Actual labels for the data
	#			Y_hat - Predicted probabilities of the labels
	#		loss - The type of loss (either square error or entropy)
	assert Y.shape == Y_hat.shape
	m = Y.shape[1]
	if (loss == "sq") :
		error = np.sum((Y-Y_hat)**2)/(2*m)
	elif (loss == "ce") :
		error = -np.sum((Y*np.log(Y_hat)))
	return error 

def back_layer(layer,cache,grads,theta,activation) :
	# Function to compute the gradient of the loss with respect to the pre-activation of a layer
	# Parameters : layer - The current layer we are acting on
	#		cache - The data stored from the forward propagation step 
	#		grads - A dictionary containing the gradients of the loss w.r.t the parameters in the network
	#		theta - A dictionary containing the weights and biases
	#		activation - The type of activation in a layer of the neural network

	dH = grads["dH"+str(layer)] # Gradient of the loss with respect to the activations of a <layer>
	A = cache["A"+str(layer)]
	H_prev = cache["H"+str(layer-1)]
	W = theta["W"+str(layer)]
	# We will be using the chain rule. It will lead to a point-wise multiplication as there is only 1 path from the pre-activation to the post-activation of a layer
	if activation=="tanh" :
		dA = dH * (1-tanh(A)**2)
	elif activation=="sigmoid" :
		dA = dH * sigmoid(A) * (1-sigmoid(A))
	
	dW = np.dot(dA,H_prev.T)
	db = np.sum(dA,axis=1)
	dH_prev = np.dot(W.T,dA)
	grads["dA"+str(layer)] = dA
	grads["dW"+str(layer)] = dW
	grads["db"+str(layer)] = db
	grads["dH"+str(layer-1)] = dH_prev

def back_prop(X,Y,Y_hat,loss,cache,grads,theta,activation,sizes) :
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
	
	# First, we need to calculate the derivative of the loss function w.r.t the output layer
	
	layers = sizes.shape[0]
	m = X.shape[1]

	if loss=="sq" :
		grads["dH"+str(layers-1)] = (Y_hat-Y)/m # The loss is the avreage loss and hence we include the <m> variable
		grads["dA"+str(layers-1)] = (Y_hat - Y) * Y_hat * (1-Y_hat)/m
	elif loss=="ce" :
		grads["dH"+str(layers-1)] = -Y / Y_hat
		grads["dA"+str(layers-1)] = -(Y-Y_hat)
	
	# We have now obtained the gradients at the output layer. We just need to backpropagate through the network to find the gradients of the loss w.r.t the parameters

	for i in range(layers-2,0,-1) :
		back_layer(i,cache,grads,theta,activation)
	
	# All of the gradients of the loss w.r.t the parameters have been added to the grads dictionary
	# Need to update the parameters after this
	return grads

def optimize(theta,grads,update,learning_rate,momentum,algo) :
	# Function to perform a certain optimization algorithm with the dictionary of gradients given
	# Parameters - theta - The dictionary of weights and biases which need to be updated
	#		grads - The dictionary of gradients which will be used to update the parameters 
	#		update - A dictionary containing the update values at instant <t-1> for the momentum based algorithm
	#		learning_rate - Rate to be used by the gradient descent algorithm; gradients will be scaled by this factor
	#		momentum - The momentum parameter for a momentum based algorithm
	#		algo - Type of optimization to be used(gradient descent, rmsprop, etc.)
	L = len(theta)/2 # Number of layers in the network excluding the input layer
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
			update["b"+str(i)] = momentum*update["W"+str(i)] + learning_rate*grads["db"+str(i)]
			theta["W"+str(i)] = theta["W"+str(i)] - update["W"+str(i)]
			theta["b"+str(i)] = theta["b"+str(i)] - update["W"+str(i)]
	elif algo == "adam" :
		i = 1

num_hidden = 3
sizes = np.array([4,3,5,2,2])
print(sizes.shape[0])
params = initialize_parameters(num_hidden,sizes)
print("W1 : "+str(params["W1"]))
#print("W4 : "+str(params["W4"]))
#print("b1 : "+str(params["b1"]))
X = np.array([[-3.34,-2.12,1.56],[0.89,2.12,0.45],[0.21,-1.98,1.56],[-0.12,0.12,1.15]])
Y = np.array([[1,0,0],[0,1,1]])
#W = params["W1"]
#b = params["b1"]
cache = {}
cache["H0"] = X
grads = {}
Y_hat = feed_forward(X,"sigmoid",params,sizes,cache)
error = cost(Y,Y_hat,"ce")
print("True Output :" + str(Y))
print("Predicted Probabilities : "+str(Y_hat))
print("Cost : " + str(error))
