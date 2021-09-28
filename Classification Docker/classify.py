from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import numpy


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

#from keras.models import load_model
model.load_weights('/content/drive/My Drive/Colab Notebooks/cats-vs-dogs/weights.h5')



from keras.preprocessing import image



#myPic = '/content/drive/My Drive/Colab Notebooks/cats-vs-dogs/data/live/what1.jpg'
myPic = '/content/drive/My Drive/Colab Notebooks/cats-vs-dogs/data/live/difficult-cat.jpg'
test_image= image.load_img(myPic, target_size = (img_width, img_height)) 
test_image = image.img_to_array(test_image)
test_image = test_image.reshape(input_shape)
test_image = numpy.expand_dims(test_image, axis = 0)
result = model.predict(test_image,verbose=0)  
print(result[0][0])

# myPic2 = '/content/drive/My Drive/Colab Notebooks/cats-vs-dogs/data/live/what2.jpg'
myPic2 = '/content/drive/My Drive/Colab Notebooks/cats-vs-dogs/data/live/corgi.jpg'
test_image2= image.load_img(myPic2, target_size = (img_width, img_height)) 
test_image2 = image.img_to_array(test_image2)
test_image2 = test_image2.reshape(input_shape)
test_image2 = numpy.expand_dims(test_image2, axis = 0)
result = model.predict(test_image2,verbose=0)  
print(result[0][0])

