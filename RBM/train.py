# Progam : Train RBM's using Contrastive Divergence algorithm
# Author : Anant Shah
# Date : 20-4-2019
# Email : anantshah200@gmail.com

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

NUM_FEATURES = 784
NUM_CLASSES = 10
NUM_HIDDEN = 100
N = 10
X_COMPONENT = 28

def get_data(train_path,test_path) :
	# Function to get the data for training
	# Augment data so as to obtain more training examples by left shifting the pixels by 1
	
	# Initialize the data
	X_train = 0
	X_test = 0

	if train_path is not None :

		train_data = pd.read_csv(train_path)
		train = np.array(train_data)
		N = train.shape[0]
		ii = np.where(train >= 127)
		io = np.where(train < 127)
		train[ii] = 1
		train[io] = 0
		X_train = train[:,1:NUM_FEATURES+1].T

		Y_train = train[:,NUM_FEATURES+1,None].T

	if test_path is not None :
		test_data = pd.read_csv(test_path)
		test = np.array(test_data)
		Y_test = test[:,NUM_FEATURES+1,None]
		ii = np.where(test[:,1:NUM_FEATURES+1] >= 127)
		io = np.where(test[:,1:NUM_FEATURES+1] < 127)
		test[ii] = 1
		test[io] = 0
		X_test = test[:,1:NUM_FEATURES+1].T
		assert X_test.shape == (NUM_FEATURES,test.shape[0])
		#print(Y_test)

	data = {"X_train" : X_train,"Y_train" : Y_train,"X_test" : X_test,"Y_test" : Y_test}

	return data

def initialize_parameters() :
	# Function to randomly initialize the parameters 
	# RBMs consist of two layers with visible variables as one layer and hidden variables as the other layer

	np.random.seed(1234)
	theta = {}
	theta["W"] = np.random.randn(NUM_HIDDEN,NUM_FEATURES) * np.sqrt(2./(NUM_FEATURES + NUM_HIDDEN)) # Xavier Initialization
	theta["b"] = np.zeros((NUM_FEATURES,1),dtype=np.float64)
	theta["c"] = np.zeros((NUM_HIDDEN,1),dtype=np.float64)	

	return theta

def sigmoid(A) :
	return 1.0/(1.0+np.exp(-A))

def train(X_train,theta,k,learning_rate,epochs) :
	# Function to train the RBM
	# Parameters : X_train - The training data
	#              theta - The parameters for the model
	#              k - The number of steps for the gibbs chain

	N = X_train.shape[1]
	W = theta["W"]
	b = theta["b"]
	c = theta["c"]
	#m = 6400
	#img_index = 1
	#fig = plt.figure()
	#fig.subplots_adjust(hspace=28,wspace=28)

	for epoch in range(epochs) :
		for i in range(1,N+1) :
			# Iterate over all the examples once for stochastic gradient 
			init_sample = X_train[:,i-1]
			init_sample = np.reshape(init_sample,(init_sample.shape[0],1))
			v_sample = X_train[:,i-1]
			v_sample = np.reshape(v_sample,(v_sample.shape[0],1))
			for t in range(k) :
				h_sample_prob = sigmoid(np.dot(W,v_sample)+c)
				h_sample = np.random.binomial(1,h_sample_prob)# Convert to 0's and 1's
				v_sample_prob = sigmoid(np.dot(W.T,h_sample)+b)
				v_sample = np.random.binomial(1,v_sample_prob)# Convert to 0's and 1's
			#if i%100==0 and i<=m and epoch==0 :
			#	img = np.reshape(v_sample,(X_COMPONENT,X_COMPONENT))
			#	ax = fig.add_subplot(8,8,img_index)
			#	ax.plot(img)
			#	img_index += 1

			W = W + learning_rate*(np.dot(sigmoid(np.dot(W,init_sample)+c),init_sample.T) - np.dot(sigmoid(np.dot(W,v_sample)+c),v_sample.T))
			b = b + learning_rate*(init_sample-v_sample)
			c = c + learning_rate*(sigmoid(np.dot(W,init_sample)+c) - sigmoid(np.dot(W,v_sample)+c))
	theta["W"] = W
	theta["b"] = b
	theta["c"] = c
	print("Training Complete")
	return theta

def get_hidden(X_test,theta) :
	# function to get the hidden representation of the test data
	W = theta["W"]
	c = theta["c"]
	hidden_prob = sigmoid(np.dot(W,X_test)+c)
	hidden_rep = np.random.binomial(1,hidden_prob)
	return hidden_rep

data = get_data("train.csv","test.csv")
X_train = data["X_train"]
X_test = data["X_test"]
Y_train = data["Y_train"]
Y_test = data["Y_test"]

for k in [1] :
	theta = initialize_parameters()
	theta = train(X_train,theta,k,learning_rate=7e-4,epochs=2)
	hidden_rep = get_hidden(X_test,theta)

# Obtain the t-SNE plot
	out_embed = TSNE(n_components=2).fit_transform(hidden_rep.T)

	cmap = plt.cm.jet
	cmaplist = [cmap(i) for i in range(cmap.N)]
	cmap = cmap.from_list('Custom cmap',cmaplist,cmap.N)
	bounds = np.linspace(0,NUM_CLASSES,NUM_CLASSES+1)

	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(1,1,1)

	scat = ax.scatter(x=out_embed[:,0],y=out_embed[:,1],c=Y_test.ravel(),cmap = cmap,alpha=0.15)
	cb = plt.colorbar(scat,spacing='proportional',ticks=bounds)
	plt.show()
