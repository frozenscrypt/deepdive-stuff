from flask import Flask, request, jsonify
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from keras.preprocessing import image

import numpy
import os



img_width, img_height = 150, 150


if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)
	
	


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
			optimizer='rmsprop',
			metrics=['accuracy'])




model.load_weights('weights.h5')


app = Flask(__name__)





@app.route("/predict", methods=["POST"])
def predict():
	# Load the Input
	global model
	data = request.files['image'] # Image input
	path = os.path.join(os.getcwd()+data.filename)
	data.save(path)

	
	# Load the model
	# model = load_model()

	test_image= image.load_img(path, target_size = (img_width, img_height))
	test_image = image.img_to_array(test_image)
	test_image = test_image.reshape(input_shape)
	test_image = numpy.expand_dims(test_image, axis = 0)
	result = model.predict(test_image,verbose=0)  
	
	if result[0][0]==0.0:
		return jsonify(output='cat')
	else:
		return jsonify(output='dog')

  
  
# Start the flask app and allow remote connections
app.run(host='0.0.0.0', port = 5000)