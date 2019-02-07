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
	parameters = {}
	for i in range(1,num_hidden+2) :
		parameters["W"+str(i)] = np.random.randn(sizes[i],sizes[i-1])
		parameters["b"+str(i)] = np.zeros((sizes[i],1))
	
	return parameters

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

def feed_forward(X,activation,parameters,sizes,cache) :
	# Function to implement the forward pass throughout the whole network
	# Parameters :  X - Input Data
	#		activation - The type of activation in the hidden layers
	#		parameters - Dictionary which contains the parameters of each fully connected layer
	#		sizes - array containing the size of each layer in the network
	#		cache - Dictionary to store the activation values required for the backward pass

	m = X.shape[1] # Number of training examples
	layers = sizes.shape[0] # Total number of layers = hidden_layers + input_layer + output_layer
	H_prev = X

	for i in range(1,layers) :
		W = parameters["W"+str(i)]
		b = parameters["b"+str(i)]
		H = forward_layer(H_prev,activation,W,b,i,cache)
		H_prev = H
	
	Y_hat = softmax(H)
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
		error = np.sum((Y-Y_hat)**2)/m
	elif (loss == "ce") :
		error = -np.sum((Y*np.log(Y_hat) + (1-Y)*np.log(1-Y_hat)))
	return error 

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
Y_hat = feed_forward(X,"sigmoid",params,sizes,cache)
error = cost(Y,Y_hat,"ce")
print("True Output :" + str(Y))
print("Predicted Probabilities : "+str(Y_hat))
print("Cost : " + str(error))
