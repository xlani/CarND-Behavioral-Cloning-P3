### code source: walkthrough project 3
### https://www.youtube.com/watch?v=rpxZ87YFg0M&t=2360s

### imports

import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

### load data

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + filename
        image = cv2.imread(current_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    correction = 0.2
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement + correction)
    measurements.append(measurement - correction)

### augment data
augmented_images = []
augmented_measurements = []

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = measurement * -1.
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

#print(len(images))
#print(len(measurements))

### convert data to numpy arrays for keras
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

#print(X_train.shape)

### model architecture // adjusted Nvidia E2E deep learning
### source: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
### added one dropout layer

model = Sequential()
model.add(Lambda(lambda x: x / 255. - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping = ((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(36, 5, 5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(48, 5, 5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

### train model

model.compile(optimizer = 'adam', loss = 'mse')
history_object = model.fit(X_train, y_train, validation_split = 0.2, \
                 shuffle = True, nb_epoch = 3, verbose = 2)

### save model

model.save('model.h5')

### print the keys contained in the history object

print(history_object.history.keys())

### plot the training and validation loss for each epoch

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
