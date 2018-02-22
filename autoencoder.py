#!/usr/bin/env python

'''
Code for Autoencoder and Denoising Autoencoder using numpy

author: Hadi Salman
email: hadicsalman@gmail.com
website: hadisalman.com
'''

__author__ = 'hadi salman'
import numpy as np
import matplotlib.pyplot as plt
import argparse

from utils.import_data import load_data
from utils.activationFunctions import sigmoidActivationFunction,tanhActivationFunction,reluActivationFunction,softmax
from utils.utilities import visualize_digit, shuffle
from utils.my_plots import plot_logs
import mnist_classifier

class autoencoder:
	def __init__(self, layers, hyper_params, activation_function, denoising=False ,logging=False,load_model=False, temp_save=True):
		self.denoising = denoising
		self.nn_input_dim = layers[0]  # input layer dimensionality
		self.nn_output_dim = layers[-1]  # output layer dimensionality
		self.load=load_model
		self.temp_save = temp_save # save weights in temporary folder
		self.layers = layers		

		self.alpha=hyper_params['learning_rate']  # learning rate for gradient descent
		self.reg_lambda=hyper_params['regularizer']  # weight decay
		self.gamma = hyper_params['momentum_weight'] # momentum weight 
		
		self.activation_function=activation_function

		self.logging=logging

		#Keeping track of the momentum of the weights the gradients of the weights for momentum calculation
		self.gradients={}
		if not load_model:
			self.weights_initialization()
		else:
			self.model = pickle.load(open(self.filename, 'rb'))
		
	
	def weights_initialization(self):
		self.model = {}
		for i, (_in, _out) in enumerate(zip(self.layers[:-1], self.layers[1:])):
			W = np.random.normal(0, 0.1, (_in, _out))
			b = np.zeros((1, _out))
			self.model.update({'W'+str(i+1):W,'b'+str(i+1):b})
			self.gradients.update({'dW'+str(i+1):0*W})


	# function for visualizing the weights of the hidden layer
	def visualize_weights(self, title):
		weights = self.model['W1']
		reshaped_weights = np.reshape(weights,(28,28,self.layers[1]))
		plt.figure(1)
		for i in range(self.layers[1]):
			ax = plt.subplot(np.sqrt(self.layers[1]),np.sqrt(self.layers[1]),i+1)	
			ax.imshow(reshaped_weights[:,:,i],cmap = 'gray')
			plt.axis('off')
		plt.pause(1)	

	def calculate_entropy_loss(self, X):
		num_examples = len(X)  # train size
		decoded,_,_ = self.forward(X)
		
		logprobs = -np.multiply(np.log(decoded),X) - np.multiply(np.log(1.0-decoded),1.0-X)		
		data_loss = np.sum(logprobs)
		return 1.*data_loss / num_examples 

	def decode(self, x):
		decoded,_,_ = self.forward(x)
		return decoded

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
		W = self.model['W'+str(len(self.layers)-1)]
		b = self.model['b'+str(len(self.layers)-1)]
		a = act_h[-1].dot(W) + b	
		decoded = self.activation_function.func(a)
		return decoded, act_h, act_a

	def backprob(self, mini_batch, noisy_mini_batch):
		# Forward propagation
		if noisy_mini_batch is None:
			X= mini_batch
		else: 
			X= noisy_mini_batch

		L=len(self.layers)-2 # number of hidden layers 
		size_miniBatch = len(X)
		decoded, act_h, act_a = self.forward(X)
		output_gradient = decoded - mini_batch

		for j in range(L+1, 0, -1):
			W = self.model['W'+str(j)]
			b = self.model['b'+str(j)]
			dW_prev = self.gradients['dW'+str(j)]# for momentum calculation

			dW = (act_h[j-1].T).dot(output_gradient)
			db = np.sum(output_gradient, axis=0, keepdims=True)
			output_gradient = output_gradient.dot(W.T) * self.activation_function.derivative(act_a[j-1]) #sigmoid_dot(act[j-1])#(1 - np.power(act[j-1], 2))
			dW += self.reg_lambda*W + self.gamma*dW_prev

			dW = dW/size_miniBatch
			db = db/size_miniBatch

			W -= self.alpha * dW
			b -= self.alpha * db
			
			self.model.update({'W'+str(j):W,'b'+str(j):b})
			self.gradients.update({'dW'+str(j):dW})		


	def train_model(self, data, mini_batch_size=1 ,num_epochs=200, print_loss=False,visualize_wegiths_while_training=False):
		# Initialize the parameters to random values. We need to learn these.
		X = data['train']['X']
		X_valid = data['valid']['X']

		train_loss_array=[]
		valid_loss_array=[]
		
		for i in range(0, num_epochs):

			N=len(X) # size of training set
			shuffled_indices = shuffle(X)
			X = X[shuffled_indices,:]
			
			X_noisy = None
			if self.denoising:
				X_noisy=self.add_noise(X)
				noisy_mini_batches = [X_noisy[k:k+mini_batch_size,:] for k in np.arange(0, N, mini_batch_size)]
			mini_batches = [X[k:k+mini_batch_size,:] for k in np.arange(0, N, mini_batch_size)]
			
			for j in range(N//mini_batch_size):
				if self.denoising:
					self.backprob(mini_batches[j], noisy_mini_batches[j]) # does backpropagation on a batch and updates weights of the model
				else:
					self.backprob(mini_batches[j], None)

			if self.logging:
				train_loss = self.calculate_entropy_loss(X)
				valid_loss = self.calculate_entropy_loss(X_valid)
				train_loss_array.append(train_loss)
				valid_loss_array.append(valid_loss)
				self.logs={'losses':{'train':train_loss_array,'valid':valid_loss_array}}

			if print_loss and i % 1 == 0:
				if visualize_wegiths_while_training:
					self.visualize_weights('epoch_'+str(i+1))
	
				if not self.logging:
					train_loss = self.calculate_entropy_loss(X)
				
				print('[Epoch: %i ...training Loss: %f]' % (i+1, train_loss))

		return self.model

	def visualize_decoded(self,x_test):
		n=10 # number of digits to test on
		ind = np.random.choice(x_test.shape[0], n)

		plt.figure(figsize=(20, 4))
		for i in range(n):

			# display original
			ax = plt.subplot(2, n, i + 1)
			plt.imshow(x_test[ind[i]].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)

			# display reconstruction
			ax = plt.subplot(2, n, i + 1 + n)
			decoded = self.decode(x_test[ind[i]])
			plt.imshow(decoded.reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
		plt.show()

	def add_noise(self, X, display=False):
		ind0 = np.random.choice(X.shape[0], int(np.floor(0.1*X.size)))
		ind1 = np.random.choice(X.shape[1], int(np.floor(0.1*X.size)))
		X_noisy = X.copy()
		X_noisy[ind0,ind1] = 0
		X_noisy = X_noisy + np.random.normal(0, 0.1,X_noisy.shape)
		
		if display:
			n=10 # number of digits to test on
			ind = np.random.choice(X.shape[0], n)

			plt.figure(figsize=(20, 4))
			for i in range(n):
				# display original
				ax = plt.subplot(2, n, i + 1)
				plt.imshow(X[ind[i]].reshape(28, 28))
				plt.gray()
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)
				
				# display reconstruction
				ax = plt.subplot(2, n, i + 1 + n)
				plt.imshow(X_noisy[ind[i]].reshape(28, 28))
				plt.gray()
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)
			plt.show()
		return X_noisy

def main(args):
	data = load_data()
	X_train = data['train']['X']
	y_train = data['train']['y']
	X_valid = data['valid']['X']
	y_valid = data['valid']['y']
	X_test = data['test']['X']
	y_test = data['test']['y']

	print("<<<<<<<<<<<<.....Executing code .....>>>>>>>>>>>>>>>")
	
	np.random.seed(0)	

	parameters={'learning_rate': args.lr, 'regularizer': args.regularizer, 'momentum_weight':args.momentum, 'mini_batch_size':args.mini_batch_size }
	
	act_function=sigmoidActivationFunction #sigmoidActivationFunction or tanhActivationFunction or reluActivationFunction  
	layers = [28*28, 100, 28*28] # specify the dimension of each layer [inputLayer, hidden1, hidden2 ...hiddenM, OutputLayer]
	
	#Initialize the network (number of layers, number of nodes in hidden layers, load weights ... )
	my_autoencoder = autoencoder(layers=layers, denoising=args.denoising, logging=True, hyper_params=parameters,activation_function=act_function, load_model=False, temp_save=True)
	
	# visualize noising of the data
	# my_autoencoder.add_noise(X_train,display=True)
	
	## train the network
	model = my_autoencoder.train_model(data, mini_batch_size=parameters['mini_batch_size'], num_epochs=args.epoch, print_loss=True, visualize_wegiths_while_training=False)	
	
	#visualization of a sample of the decoded images and their corresoinding input images
	my_autoencoder.visualize_decoded(X_test)

	#plot the loss vs the number of epochs
	plot_logs(my_autoencoder.logs)
	# ----------------------------------------------------------------------------------------------------------------------------------
	if args.train_classifier:
		print('Starting the training of the MLP classifier on the MNIST dataset!')
		# MNIST-network Unsupervised Pretaraining
		#Network configurations 
		parameters={'learning_rate': 0.1, 'regularizer': 0, 'momentum_weight':0, 'mini_batch_size':100 }
		
		act_function=reluActivationFunction #sigmoidActivationFunction or tanhActivationFunction or reluActivationFunction  
		layers = [28*28,100,10] # specify the dimension of each layer [inputLayer, hidden1, hidden2 ...hiddenM, OutputLayer]
		
		#Initialize the network (number of layers, number of nodes in hidden layers, load weights ... )
		network = mnist_classifier.mnist_network(layers=layers, logging=True, hyper_params=parameters, activation_function=act_function)
		network.unsupervised_pretraining(my_autoencoder)
		network.visualize_weights('Weights before Training')# visualize initial weights
		
		## train the network
		model = network.train_model(data, mini_batch_size=parameters['mini_batch_size'], 
							num_epochs=200, print_loss=True, visualize_wegiths_while_training=False)
		network.visualize_weights('Weights after Training')# visualize weights after training

		# plots accuracies, entropy losses and mean classification error as a function of number of epochs for training and validation datasets	
		plot_logs(network.logs)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--lr', type=float, default=0.2, help='Learning rate')
	parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
	parser.add_argument('--regularizer', type=float, default=0.001, help='regularizer')
	parser.add_argument('--mini-batch-size', type=int, default=100, help='Minibatch size for RBM training')
	parser.add_argument('--epoch', type=int, default=20, help='Number of epochs to train')
	parser.add_argument('--denoising', action='store_true', help='Trains an MLP calissifier with unsupervised pretraining')
	parser.add_argument('--train_classifier', action='store_true', help='Trains an MLP calissifier with unsupervised pretraining')

	args = parser.parse_args()

	main(args=args)