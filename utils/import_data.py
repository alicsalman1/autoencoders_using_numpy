import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def visualize_digit(X):
	image = np.reshape(X,(28,28))
	plt.imshow(image)
	plt.show()

def load_data():
	data_dir = 'data/'
 
	data_train = np.genfromtxt(os.path.join(data_dir,'digitstrain.txt'),delimiter=',')
	data_valid = np.genfromtxt(os.path.join(data_dir,'digitsvalid.txt'),delimiter=',')
	data_test = np.genfromtxt(os.path.join(data_dir,'digitstest.txt'),delimiter=',')

	#training data
	X_train = data_train[:,:-1]
	y_train = data_train[:,-1]
	#validation data
	X_valid = data_valid[:,:-1]
	y_valid = data_valid[:,-1]
	#testing data
	X_test = data_test[:,:-1]
	y_test = data_test[:,-1]

	return{'train':{'X':X_train,'y':y_train},'valid':{'X':X_valid,'y':y_valid},'test':{'X':X_test,'y':y_test}}

if __name__ == "__main__":

	data = load_data()
	X = data['train']['X']
