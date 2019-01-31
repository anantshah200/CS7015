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
	"Function to initialize the arguments
	 Parameters : num_hidden - Number of hidden layers 
	 	sizes - a list with number of perceptrons in each hidden layer"
	np.random.seed(1234)
	parameters = {}
	for i in range(1,num_hidden+1) :
		parameters["W"+str(i)] = np.random.randn(sizes[i],sizes[i-1])
		parameters["b"+str(i)] = 0.0
