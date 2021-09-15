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
from pickle import dump
import sys
import json



def plot_metrics(history,img_dir):

	'''
		Args:
			history: history object out of keras model fit, storing information on training 
			img_dir: directory to save plots
	'''

	train_loss = history.history['loss']
	train_acc = history.history['accuracy']
	val_loss = None
	val_acc = None
	if 'val_loss' in history.history:
		val_loss = history.history['val_loss']
		val_acc = history.history['val_accuracy']


	plt.figure(figsize=(15,10))
	plt.plot(range(len(train_loss)), train_loss, 'r', "Training Loss")
	if val_loss:
		plt.plot(range(len(val_loss)), val_loss, 'b', "Validation Loss")
	plt.title('Loss')
	plt.savefig(os.path.join(img_dir,'loss.png'))


	plt.figure(figsize=(15,10))
	plt.plot(range(len(train_acc)), train_acc, 'r', "Training Accuracy")
	if val_acc:
		plt.plot(range(len(val_acc)), val_acc, 'b', "Validation Accuracy")
	plt.title('Accuracy')
	plt.savefig(os.path.join(img_dir,'accuracy.png'))







''' Class for Model Definition'''

class Model(Sequential):


	'''
	Defines a Sequential model inherited out of keras.models

	'''

	def __init__(self, n_steps=7, lstm_dim=200, n_classes=22):
		Sequential.__init__(self)
		objective = 'softmax' if n_classes>2 else 'sigmoid'
		self.add(Input(shape=(n_steps,4)))
		self.add(LSTM(lstm_dim,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), return_sequences=True))#1e-5,1e-4, 150
		self.add(BatchNormalization())
		self.add(LSTM(150,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))#1e-5,1e-4, 150
		self.add(BatchNormalization())
		self.add(Dense(60))
		self.add(keras.layers.Dropout(0.3))
		self.add(BatchNormalization())
		self.add(ReLU())
		self.add(Dense(n_classes,activation=objective))



''' Class for Training and Test Data preparation. The class reads data from all participant csvs in the given folder and splits them into train and
 test sets by randomly sampling a fraction of data from each class thereby containing same proportion of each class in train and test set'''

class DataPreparation(object):

	'''
		Args:
			dir_path: path to the directory storing all csvs. The csvs should be named as 1.csv, 2.csv and so on
			train_path: path to store training data as csv
			test_path: path to store test data as csv
			scaler_path: path to store StandardScaler model of scikit-learn in pkl file
			n_classes: number of classes to predict
			test_size: size of test data. Should be between 0 and 1
			n_steps: sequence length of each sample

	'''
	def __init__(self, dir_path, train_path=None, test_path=None, scaler_path = None, n_classes=22, test_size=0.1, n_steps = 7):
		assert test_size>=0.0 and test_size<=1.0
		self.dir_path = dir_path
		self.n_classes = n_classes
		self.test_size = test_size
		self.train_path = train_path
		self.test_path = test_path
		self.n_steps = n_steps
		self.scaler_path = scaler_path


	'''
		Output:
			Numpy arrays of train_X, train_y, test_X, test_y
	'''

	def get_data(self):
		raw_data = []
		lengths = []
		test_data = []
		df_lens = []
		test_lens = []
		j = 0
		for i in range(1,self.n_classes+1):

			data = pd.read_csv(os.path.join(self.dir_path,'{}.csv'.format(i)),names=['timestamp','x_acceleration','y_acceleration','z_acceleration'])

			timestamps = data.timestamp.iloc[:]
			diff = [0.0]+[timestamps[i]-timestamps[i-1] for  i in range(1,len(timestamps))]
			data['time_diffs'] = diff
			data['labels'] = [j]*len(data)


			ind = np.random.randint(len(data)-int(0.1*len(data)))
			test = data.loc[ind:ind+int(0.1*len(data)),:]
			test_lens.append(len(test))

			df1 = data.loc[:ind-1,:]
			df2 = data.loc[ind+int(0.1*len(data))+1:,:]

			df_lens.append(len(df1))
			temp = pd.concat([df1,df2], axis=0)
			lengths.append(len(temp))

			if len(raw_data)==0:
				raw_data = temp
				test_data = test
			else:
				raw_data = pd.concat([raw_data,temp],axis=0)
				test_data = pd.concat([test_data,test],axis=0)
			j+=1



		scaler = StandardScaler()
		scaler.fit(raw_data.loc[:,['x_acceleration', 'y_acceleration', 'z_acceleration', 'time_diffs']])
		raw_data_std = scaler.transform(raw_data.loc[:,['x_acceleration', 'y_acceleration', 'z_acceleration', 'time_diffs']])

		test_data_std = scaler.transform(test_data.loc[:,['x_acceleration', 'y_acceleration', 'z_acceleration', 'time_diffs']])

		
		dataset = []
		labels = []
		test_set = []
		test_labels = []
		ind = 0
		tind = 0
		for i in range(len(lengths)):
			data = raw_data_std[ind:ind+lengths[i],:]
			test = test_data_std[tind:tind+test_lens[i],:]

			ind+=lengths[i]
			tind+=test_lens[i]

			df1 = data[:df_lens[i],:]
			df2 = data[df_lens[i]+1:,:]
			for dt in [df1,df2]:
				for k in range(0,len(dt)-self.n_steps):
					sub_df = dt[k:k+self.n_steps,:]
					dataset.append(np.array(sub_df))
					labels.append(i)


			for k in range(0,len(test)-self.n_steps):
				sub_df = test[k:k+self.n_steps,:]
				test_set.append(np.array(sub_df))
				test_labels.append(i)

			
		final_data = np.array(dataset)
		final_labels = np.array(labels)
		test_set = np.array(test_set)
		test_labels = np.array(test_labels)

		if self.n_classes>2:
			final_catlabels = tf.keras.utils.to_categorical(final_labels,num_classes=self.n_classes)
			test_labels = tf.keras.utils.to_categorical(test_labels,num_classes=self.n_classes)
		else:
			final_catlabels = final_labels

		final_data,final_catlabels = shuffle(final_data,final_catlabels)
		test_set,test_labels = shuffle(test_set,test_labels)


		if self.scaler_path:
			dump(scaler,open(self.scaler_path,'wb'))

		if self.train_path:
			raw_data.to_csv(self.train_path)

		if self.train_path:
			test_data.to_csv(self.test_path)


		return final_data, final_catlabels, test_set, test_labels



if __name__=='__main__':

	config = json.load(open(sys.argv[1],'r'))
	dir_path = config['dir_path']
	train_path = config['train_path']
	test_path = config['test_path']
	scaler_path = config['scaler_path']
	model_path = config['model_path']
	n_classes = int(config['n_classes'])
	test_size = float(config['test_size'])
	n_steps = int(config['n_steps'])
	lstm_dim = int(config['lstm_dim'])
	n_epochs = int(config['n_epochs'])
	img_dir = config['plot_path']


	dp = DataPreparation(dir_path=dir_path, train_path=train_path, test_path=test_path, scaler_path=scaler_path, n_classes=n_classes, test_size=test_size,
						n_steps=n_steps)

	X_train, y_train, X_test, y_test = dp.get_data()
	print(X_train.shape,y_train.shape)
	
	model = Model(lstm_dim=lstm_dim, n_classes=n_classes, n_steps=n_steps)
	loss = 'categorical_crossentropy' if n_classes>2 else 'binary_crossentropy'
	# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
	# 	1e-3,
	# 	decay_steps=167600,
	# 	decay_rate=0.96,
	# 	staircase=True)

	opt = optimizers.Adam(learning_rate=1e-3)#3,4
	
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


	history = model.fit(X_train,y_train,shuffle=True,batch_size=64,epochs=n_epochs, verbose=2, validation_split=0.2)
	model.save(model_path)
	plot_metrics(history,img_dir)

	loss, test_accuracy = model.evaluate(X_test, y_test)

	print("Accuracy on Test Set = ", test_accuracy)









				




