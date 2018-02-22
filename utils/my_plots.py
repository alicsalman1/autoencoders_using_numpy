import matplotlib.pyplot as plt

def plot_logs(logs):
	if 'accuracies' in logs.keys():	
		train_acc_array=logs['accuracies']['train']#training accuracy
		valid_acc_array=logs['accuracies']['valid']#validation accuracy
	if 'losses' in logs.keys():		
		train_loss_array=logs['losses']['train'] # average entropy loss
		valid_loss_array=logs['losses']['valid'] # average entropy loss
	if 'mean_classification_error' in logs.keys():		
		train_classification_loss_array=logs['mean_classification_error']['train']# mean classification loss
		valid_classification_loss_array=logs['mean_classification_error']['valid']# mean classification loss
	
	num_epochs = len(logs[list(logs.keys())[0]]['train'])
	
	x = range(1,num_epochs+1)
	n = len(logs.keys())
	f, axarr = plt.subplots(n, sharex=True)
	i=0
	if n==1:
		axarr=[axarr]
	
	if 'accuracies' in logs.keys():	
		axarr[i].plot(x, train_acc_array,color='r',label='train')
		axarr[i].plot(x, valid_acc_array,color='b',label='valid')
		axarr[i].set_title('Accuracies vs. Number of Epochs')
		i=i+1

	if 'losses' in logs.keys():			
		axarr[i].plot(x, train_loss_array,color='r',label='train')
		axarr[i].plot(x, valid_loss_array,color='b',label='valid')
		axarr[i].set_title('Average Entropy Losses vs. Number of Epochs')
		i=i+1

	if 'mean_classification_error' in logs.keys():			
		axarr[i].plot(x, train_classification_loss_array,color='r',label='train')
		axarr[i].plot(x, valid_classification_loss_array,color='b',label='valid')
		axarr[i].set_title('mean classification errors vs. Number of Epochs')
		
	plt.legend()
	plt.show()


