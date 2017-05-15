import numpy as np

import keras
import config as cnf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.optimizers import SGD
from keras import regularizers
from images import load_tile
import matplotlib.pyplot as plt

from numpy import genfromtxt
marking = genfromtxt("training/marking.txt", dtype='int',delimiter=',')

data = np.zeros((marking.shape[0],cnf.square_size, cnf.square_size, 3), dtype=np.float)



for x in range(0,marking.shape[0]):
	im = load_tile(marking[x][0],marking[x][1])
	data[x] = im

x_train = data[0:cnf.training_count:1]   #обучающее множество
x_test  = data[cnf.training_count:100:1] #проверочное множество

y_train = keras.utils.to_categorical(marking[0:cnf.training_count,2], num_classes=5)
y_test = keras.utils.to_categorical(marking[cnf.training_count:100,2], num_classes=5)

model = Sequential()
# Картинки 150x150 с тремя каналами -> Тензоры (150, 150, 3).

model.add(BatchNormalization(input_shape=(150, 150, 3))) #первый слой - нормализация
                                                         #в первом слое указывают input_shape
model.add(Conv2D(32, (3, 3), activation='relu')) #свертка 3x3, 32 типа фич.
model.add(Conv2D(32, (3, 3), activation='relu')) #свертка
model.add(MaxPooling2D(pool_size=(2, 2))) #макспулинг
model.add(Dropout(0.25)) #отбрасывается каждый четвертый выход (метод борьбы с переобучением)

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Flatten()) #совмещение трёх слоев (надо проверить как оно точно происходит)
model.add(Dense(512, kernel_regularizer=regularizers.l2(0.01), activation='relu')) #связный слой с 512 нейронами и регуляризацией
model.add(Dropout(0.25))
model.add(Dense(5, activation='softmax'))#выходной слой

sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=20, epochs=400)
#score = model.evaluate(x_test, y_test, batch_size=32)
predict_train = model.predict(x_train,verbose=1);
predict_test = model.predict(x_test,verbose=1);
print(predict_train)
print(predict_test)
print(model.predict_classes(x_train))
print(marking[0:cnf.training_count,2])
print(model.predict_classes(x_test))
print(marking[cnf.training_count:100,2])
# summarize history for accuracy
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()