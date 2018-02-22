import numpy as np

class linearActivationFunction(object):
	@staticmethod
	def func(x):
	  return x

	@staticmethod  
	def derivative(x):
		"""Derivative of the linear function."""
		return np.zeros(x.shape)

class reluActivationFunction(object):
	@staticmethod
	def func(x):
		out=x
		out[x<0]=0
		return out 

	@staticmethod  
	def derivative(x):
		"""Derivative of the relu function."""
		return np.asarray((x>0),np.float) 

class tanhActivationFunction(object):
	@staticmethod
	def func(x):
	  return np.tanh(x)

	@staticmethod  
	def derivative(x):
		"""Derivative of the tanh function."""
		return 1-np.power(np.tanh(x), 2) 


class sigmoidActivationFunction(object):
	@staticmethod
	def func(x):
	  return 1./ (1 + np.exp(-x))

	@staticmethod  
	def derivative(x):
		"""Derivative of the sigmoid function."""
		sigmoid = 1./ (1 + np.exp(-x))
		return sigmoid*(1-sigmoid)

def softmax(x):
	exp_scores = np.exp(x)
	return 1.*exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
