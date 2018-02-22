import numpy as np
import random

def visualize_digit(X):
	image = np.reshape(X,(28,28))
	plt.imshow(image)
	plt.show()

# shuffles a list
def shuffle(X):
	index_shuf = np.arange(X.shape[0])
	random.shuffle(index_shuf)

	return index_shuf
