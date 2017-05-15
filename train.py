import numpy as np

import keras
import config as cnf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.optimizers import SGD
from images import load_tile

from numpy import genfromtxt
marking = genfromtxt("training/marking.txt", dtype='int',delimiter=',')

data = np.zeros((marking.shape[0],cnf.square_size, cnf.square_size, 3), dtype=np.float)

for x in range(0,marking.shape[0]):
	im = load_tile(marking[x][0],marking[x][1])
	data[x] = im

x_train = data[0:80:1]
x_test  = data[80:100:1]

y_train = keras.utils.to_categorical(marking[0:80,2], num_classes=5)
y_test = keras.utils.to_categorical(marking[80:100,2], num_classes=5)

model = Sequential()
# Картинки 150x150 с тремя каналами -> Тензоры (150, 150, 3).

model.add(BatchNormalization(input_shape=(150, 150, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=40, epochs=80)
#score = model.evaluate(x_test, y_test, batch_size=32)
predict_train = model.predict(x_train,verbose=1);
predict_test = model.predict(x_test,verbose=1);
print(predict_train)
print(predict_test)
print(model.predict_classes(x_train))
print(model.predict_classes(x_test))