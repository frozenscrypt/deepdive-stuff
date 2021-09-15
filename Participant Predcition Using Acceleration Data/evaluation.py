import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, Dense, BatchNormalization, ReLU, LSTM
from keras import optimizers
import tensorflow as tf
import keras
from tensorflow.keras import regularizers
import os
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from pickle import load
import sys
import json



def evaluate(test_path, model_path, scaler_path, n_classes):

	'''
		Args:
			test_path: path to test data csv
			model_path: path to the saved model
			scaler_path: path to the saved StandardScaler pkl file
			n_classes: number of classes to predict from
	'''

	model = keras.models.load_model(model_path)

	data = pd.read_csv(test_path)
	
	scaler = load(open(scaler_path, 'rb'))

	data_std = scaler.transform(data.loc[:,['x_acceleration','y_acceleration','z_acceleration','time_diffs']])
	labels = data.loc[:,'labels']
	

	i = 0
	ind = 0
	label_counts = dict(labels.value_counts())
	test_set = []
	test_labels = []
	while i<len(label_counts):
		df = data_std[ind:ind+label_counts[i],:]
		ind+=label_counts[i]
		for k in range(0,len(df)-n_steps):
			sub_df = df[k:k+n_steps,:]
			test_set.append(np.array(sub_df))
			test_labels.append(i)
		i+=1

	test_set = np.array(test_set)
	test_labels = np.array(test_labels)

	if n_classes>2:
		test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=n_classes)

	loss,accuracy = model.evaluate(test_set, test_labels)

	print("Test set accuracy = ",accuracy)

	return loss, accuracy



if __name__=='__main__':
	config = json.load(open(sys.argv[1],'r'))
	test_path = config['test_path']
	model_path = config['model_path']
	scaler_path = config['scaler_path']
	n_steps = config['n_steps']
	n_classes = config['n_classes']
	
	loss, accuracy = evaluate(test_path, model_path, scaler_path, n_classes)


