#
# Behavioral Cloning implementation
# ppradhan01@gmail.com
#

import csv
import cv2
import numpy as np

# get image files and measurements to build X and y
r = 0
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        # skip header
        if r==0:
            r+=1
            continue
        lines.append(line)

print('read %s records' % len(lines))

# get actual image data
# augment and preprocess data set
images = []
measurements = []
# steering correction parameter to simulate
# recovery driving using left/right cameras
correction = 0.25
for line in lines:

    # center image and steering
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    img = cv2.imread(current_path)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

    # augment with flipped image (only for center image)
    image_flipped = np.fliplr(image)
    images.append(image_flipped)
    measurement_flipped = -measurement
    measurements.append(measurement_flipped)

    # augment with left camera image to simulate recovery from left edge
    source_path = line[1]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    img = cv2.imread(current_path)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    measurement_left = measurement + correction
    images.append(image)
    measurements.append(measurement_left)

    # augment with right camera image to simulate recovery from right edge
    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    img = cv2.imread(current_path)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    measurement_right = measurement - correction
    images.append(image)
    measurements.append(measurement_right)

print('read %s records' % len(images))

# not using generator - final data size (4X) seems ok for memory
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))

# final - NVIDIA
model.add(Convolution2D(24,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(36,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(48,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(MaxPooling2D())
# skipping one convolutional layer from NVIDIA model
model.add(Flatten())
model.add(Dense(1164))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('model.h5')
