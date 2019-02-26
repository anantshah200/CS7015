import numpy as np
import pandas as pd
import csv
from sklearn.metrics import accuracy_score

layers = 5
test = {}
#test["A"+str(layers)] = 4
#print("A"+str(layers))

for i in range(1,6) :
	print(i)

def create_mini_batch(X, Y, batch_size) :
	N = X.shape[1]
	indices = np.random.permutation(N)
	mini_batch_indices = [] # A list containing each set of indices
	num_batches = N / batch_size
	for i in range(int(num_batches)):
		mini_batch_indices.append(indices[i*batch_size:(i+1)*batch_size])

	return mini_batch_indices

X = np.ones((5,10))
Y = np.zeros((5,10))
batch_size = 5
sizes = create_mini_batch(X, Y, batch_size)

#for index in sizes :
#	X_batch = X[:,index]
#	Y_batch = Y[:,index]
#	print("X array :" + str(X_batch))
#	print("Y array :" + str(Y_batch))

y_0 = np.zeros((1,5)).astype(int)
y_1 = y_0 + 1
y_2 = y_0 + 2
y_3 = y_0 + 3

Y = np.concatenate((y_0,y_1,y_2,y_3),axis=1)
index_1 = np.where(Y[0][:] == 1)
index_2 = np.where(Y[0][:] == 2)
index_3 = np.where(Y[0][:] == 3)
#print(index_1)
#print(index_2)
#print(index_3)

train = "dl2019pa1/train.csv"
def readcsv(train) :
	xfile = open(train,'r')
	data = csv.reader(xfile,delimiter=",")
	X = []
	for row in data :
		X.append(row)
	return np.array(X)

def readpanda(train) :
	data = pd.read_csv(train)
	return np.array(data)

#X_train = readpanda(train)
#print(X_train.shape)

def test_accuracy(Y_hat) :
	N = 4
	max_in = 0
	for i in range(N) :
		max_ind = np.where(Y_hat[:,i] == np.amax(Y_hat[:,i]))
		min_ind = np.where(Y_hat[:,i] != np.amax(Y_hat[:,i]))
		if(len(max_ind[0]) >= 2) :
			max_in = max_ind[0][0]
			#for j in range(1,len(max_ind[0])) :
			min_ind = np.append(min_ind[0],max_ind[0][1:])
		else :
			max_in = max_ind[0]
		Y_hat[max_in,i] = 1
		Y_hat[min_ind,i] = 0
	return Y_hat

A = np.array([[1, 2, 3],[4, 5, 6], [7, 8, 9]])
sum_A = np.sum(A,axis=0)


def softmax(A) :
	# Given a matrix of values, this function will return the softmax values
	m = A.shape[0]
	n = A.shape[1]
	A_exp = np.exp(A) # Found the exponential of all the values
	e_sum = np.sum(A_exp,axis=0,keepdims=True)
	A_softmax = A_exp/e_sum
	assert A_softmax.shape == (m,n)
	return A_softmax

def test_accuracy(Y_test,Y_hat) :
	# Function to test the accuracy of the trained model
	N = 3
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
	print(Y_hat)
	print(Y_test)
	accuracy = accuracy_score(Y_test,Y_hat)
	return accuracy

A = np.array([[0.2, 0.6, 0.1, 0.3],[0.4, 0.2, 0.8, 0.5],[0.4, 0.2, 0.1, 0.2]])
A = softmax(A)
A_sum = np.sum(A,axis=0,keepdims=True)
print(A_sum)
B = np.array([[0.2, 0.4, 0.4], [0.4, 0.1, 0.5], [0.4, 0.5, 0.1]])
B_true = np.array([[0, 0, 0],[1,0,1],[0,1,0]])
acc = test_accuracy(B_true,B)
print(acc)

A = {"A_1" : 1, "B_1" : 2}
B = {"A_2" : 3, "B_2" : 4}
params = {"A" : A.copy(), "B" : B.copy()}
A["A_1"] = 5
A["B_1"] = 6
print(params)

A = np.array([[1, 5], [2, 6]])
cost = np.sum(A,axis=1,keepdims=True)
print(cost)
cost = np.squeeze(cost)
print(cost)

N = 4
a = 1./N
print(a)

def test_model(X_test) :
	# Function to predict the outcomes of the test data
	#Y_hat, trash = feed_forward(X_train,activation,theta,sizes)
	N = X_test.shape[1]
	Y_hat = X_test
	max_val = 0
	predictions = np.zeros((N,1)).astype(int)
	
	for i in range(N) :
		max_ind = np.where(Y_hat[:,i] == np.amax(Y_hat[:,i]))
		if (len(max_ind[0]) >= 2) :
			max_val = max_ind[0][0]
		else :
			max_val = np.squeeze(max_ind[0])
		predictions[i] = max_val
	print(predictions.shape)
	index = (np.arange(0,N).T).reshape(N,1)
	print(index)
	print(index.shape)
	predictions = np.hstack((index,predictions))
	return predictions
X_test = np.array([[0.2, 0.4, 0.8, 0.3],[0.3, 0.1, 0.1, 0.1],[0.5, 0.5, 0.1, 0.6]])
a = test_model(X_test)
pd.DataFrame(a).to_csv("submission.csv", header=["id","label"],index=False)
print(a)

A = np.array([[1,0,0],[0,0,1],[0,1,0]])
B = np.array([[0.9,0.8,0.7],[0.6,0.5,0.4],[0.3,0.2,0.1]])
print(A)
ind = np.where(A.T==1)
print(ind)
C = B[ind[1],ind[0]]
C = np.tile(C,(3,1))
print(C)

D = np.array([[[1,1,1,1],[2,2,2,2],[3,3,3,3],[0,0,0,0]],[[4,4,4,4],[5,5,5,5],[6,6,6,6],[9,9,9,9]],[[7,7,7,7],[8,8,8,8],[9,9,9,9],[9,9,0,0]]])
print(D.shape)
np.random.seed(1)
E = np.random.randn(16,5)
print(E)
F = np.reshape(E.T,(5,4,4),order='C')
print("F :"+str(F))
G = np.roll(F[:],1,axis=1)
#G = np.reshape(G,(5,16))
print("G : "+str(G))
H = np.reshape(G,(5,16))
print("H "+str(H.T))
#print(sum_A)
#print(np.exp(A))
#print("Sizes : " + str(sizes))
