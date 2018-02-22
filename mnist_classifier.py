#!/usr/bin/env python
'''
author: Hadi Salman
email: hadicsalman@gmail.com
website: hadisalman.com
'''
import numpy as np
import matplotlib.pyplot as plt

from utils.import_data import load_data
from utils.activationFunctions import sigmoidActivationFunction,tanhActivationFunction,reluActivationFunction,softmax
from utils.utilities import visualize_digit, shuffle
from utils.my_plots import plot_logs

class mnist_network:
	def __init__(self,layers, hyper_params, activation_function, logging=False):
		self.nn_input_dim = layers[0]  # input layer dimensionality
		self.nn_output_dim = layers[-1]  # output layer dimensionality
		self.layers = layers
		
		self.alpha=hyper_params['learning_rate']  # learning rate for gradient descent
		self.reg_lambda=hyper_params['regularizer']  # weight decay
		self.gamma = hyper_params['momentum_weight'] # momentum weight 
		
		self.activation_function=activation_function

		self.logging=logging

		#Keeping track of the gradients of the weights for momentum calculation
		self.gradients={}		
		self.weights_initialization()
		
	def unsupervised_pretraining(self, pretrained_model):
		if 'W1' in pretrained_model.model.keys():
			self.model['W1'] = 	pretrained_model.model['W1']
		elif 'W' in pretrained_model.model.keys():
			self.model['W1'] = 	pretrained_model.model['W']

	def weights_initialization(self):
		self.model = {}
		# np.random.seed(0)	
		for i, (_in, _out) in enumerate(zip(self.layers[:-1], self.layers[1:])):
			bound = np.sqrt(6)/np.sqrt(_in + _out)
			W = bound*np.random.uniform(-1,1,(_in, _out))
			b = np.zeros((1, _out))
			self.model.update({'W'+str(i+1):W,'b'+str(i+1):b})
			self.gradients.update({'dW'+str(i+1):0*W})
		return

	# function for visualizing the weights of the hidden layer
	def visualize_weights(self, title):
		weights = self.model['W1']
		reshaped_weights = np.reshape(weights,(28,28,self.layers[1]))
		plt.figure(1)
		print(title)
		for i in range(self.layers[1]):
			ax = plt.subplot(np.sqrt(self.layers[1]),np.sqrt(self.layers[1]),i+1)	
			ax.imshow(reshaped_weights[:,:,i],cmap='gray')
			plt.axis('off')
		plt.show()
	
		return

	# function to test the network on the validation and test sets to get accuracy
	def test(self,X_test,y_true):
		y_hat=self.predict(X_test)
		accuracy = 1.*np.sum(y_hat == y_true)/len(y_hat)*100
		return accuracy

	def calculate_classification_loss(self, X, y):
		y_hat=self.predict(X)
		classification_loss = 1.*np.sum(y_hat != y)/len(y_hat)
		
		return classification_loss 

	def calculate_entropy_loss(self, X, y):
		num_examples = len(X)  # size of training set 
		probs,_,_ = self.forward(X)
		
		# Calculating the cross-entropy loss
		logprobs = -np.log(probs[range(num_examples), y.astype(int)])
		data_loss = np.sum(logprobs)

		return 1.*data_loss / num_examples 


	def predict(self, x):
		probs,_,_ = self.forward(x)
		return np.argmax(probs, axis=1)

	def forward(self, x):
		L=len(self.layers)-2 # number of hidden layers 
		act_h=[x]
		act_a=[x]
		# Forward propagation
		for j in range(L):
			W = self.model['W'+str(j+1)]
			b = self.model['b'+str(j+1)]

			a = act_h[-1].dot(W) + b
			h=self.activation_function.func(a)

			act_a.append(a)
			act_h.append(h)
		# embed()
		W = self.model['W'+str(len(self.layers)-1)]
		b = self.model['b'+str(len(self.layers)-1)]
		a = act_h[-1].dot(W) + b	
		probs = softmax(a)

		return probs, act_h, act_a

	def backprob(self, mini_batch):
		# Forward propagation
		L=len(self.layers)-2 # number of hidden layers 

		X,y = mini_batch
		size_miniBatch = len(X)
		probs, act_h, act_a = self.forward(X)
		# Backpropagation
		output_gradient = probs
		output_gradient[range(size_miniBatch), y.astype(int)] -= 1

		for j in range(L+1, 0, -1):
			W = self.model['W'+str(j)]
			b = self.model['b'+str(j)]
			dW_prev = self.gradients['dW'+str(j)]# for momentum calculation

			dW = (act_h[j-1].T).dot(output_gradient)
			db = np.sum(output_gradient, axis=0, keepdims=True)
			output_gradient = output_gradient.dot(W.T) * self.activation_function.derivative(act_a[j-1])
			
			#regularization + momentum
			dW += self.reg_lambda*W + self.gamma*dW_prev

			dW = dW/size_miniBatch
			db = db/size_miniBatch

			W -= self.alpha * dW
			b -= self.alpha * db
			self.model.update({'W'+str(j):W,'b'+str(j):b})
			self.gradients.update({'dW'+str(j):dW})
		return
		
	# This function learns parameters for the neural network and returns the model.
	# - nn_hdim: Number of nodes in the hidden layer
	# - num_epochs: Number of epochs
	# - print_loss: If True, print the loss 10 epochs
	def train_model(self, data, mini_batch_size=1 ,num_epochs=200, print_loss=False,visualize_wegiths_while_training=False):
		# Initialize the parameters to random values. We need to learn these.
		X = data['train']['X']
		y = data['train']['y']

		X_valid = data['valid']['X']
		y_valid = data['valid']['y']

		X_test = data['test']['X']
		y_test = data['test']['y']
		
		train_acc_array=[]
		valid_acc_array=[]
		train_loss_array=[]
		valid_loss_array=[]
		train_classification_loss_array=[]
		valid_classification_loss_array=[]
		
		for i in range(0, num_epochs):
			
			N=len(X) # size of training set
			shuffled_indices = shuffle(X)

			X = X[shuffled_indices,:]
			y = y[shuffled_indices]
			training_data = zip(X,y)
			
			mini_batches = [[X[k:k+mini_batch_size,:],y[k:k+mini_batch_size]] for k in np.arange(0, N, mini_batch_size)]
			for j in range(N//mini_batch_size):
				self.backprob(mini_batches[j]) # does backpropagation on a batch and updates weights of the model
	
			if self.logging:
				train_acc = self.test(X,y)
				valid_acc = self.test(X_valid,y_valid)
				train_loss = self.calculate_entropy_loss(X, y)
				valid_loss = self.calculate_entropy_loss(X_valid, y_valid)
				train_classification_loss = self.calculate_classification_loss(X, y)
				valid_classification_loss = self.calculate_classification_loss(X_valid, y_valid)
		
				## save the accuracies and losses
				train_acc_array.append(train_acc)
				valid_acc_array.append(valid_acc)
				train_loss_array.append(train_loss)
				valid_loss_array.append(valid_loss)
				train_classification_loss_array.append(train_classification_loss)
				valid_classification_loss_array.append(valid_classification_loss)
				self.logs={'accuracies':{'train':train_acc_array,'valid':valid_acc_array},'losses':{'train':train_loss_array,'valid':valid_loss_array},
									'mean_classification_error':{'train':train_classification_loss_array,'valid':valid_classification_loss_array}}

			if print_loss and i % 10 == 0:
				if visualize_wegiths_while_training:
					self.visualize_weights('epoch_'+str(i+1))
	
				if not self.logging:
					train_acc = self.test(X,y)
					valid_acc = self.test(X_valid,y_valid)
					train_loss = self.calculate_entropy_loss(X, y)
				
				print('[Epoch: %i .....Training accuracy: %f ...Validation accuracy: %f...training Loss: %f]' % (i, train_acc, valid_acc, train_loss))

		return self.model


def main():
	data = load_data()
	X_train = data['train']['X']
	y_train = data['train']['y']

	X_valid = data['valid']['X']
	y_valid = data['valid']['y']

	X_test = data['test']['X']
	y_test = data['test']['y']
			
	print("<<<<<<<<<<<<.....Executing code .....>>>>>>>>>>>>>>>")
	
	np.random.seed(0)	
	
	#Network configurations 
	layers = [28*28,100,10] # specify the dimension of each layer [inputLayer, hidden1, hidden2 ...hiddenM, OutputLayer]
	act_function=reluActivationFunction #sigmoidActivationFunction or tanhActivationFunction or reluActivationFunction  
	parameters={'learning_rate': 0.3, 'regularizer': 0.001, 'momentum_weight':0.9, 'mini_batch_size':32 }
	
	#Initialize the network (number of layers, number of nodes in hidden layers etc.)
	network = mnist_network(layers=layers, logging=True, hyper_params=parameters,activation_function=act_function)
	network.visualize_weights('Weights before Training')# visualize initial weights
	
	## train the network
	model = network.train_model(data, mini_batch_size=parameters['mini_batch_size'], num_epochs=100, print_loss=True, visualize_wegiths_while_training=False)
	network.visualize_weights('Weights after Training')# visualize weights after training

	# plots accuracies, entropy losses and mean classification error as a function of number of epochs for training and validation datasets	
	plot_logs(network.logs)

if __name__ == "__main__":
	main()