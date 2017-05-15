import numpy as np

import keras
import config as cnf
from keras.optimizers import SGD
from images import load_tile
import matplotlib.pyplot as plt
from map_utils import TileDataProvider
from model import init_model
from numpy import genfromtxt
import sys

if(len(sys.argv)>1 and sys.argv[1]=="train"):
    marking = genfromtxt("training/marking.txt", dtype='int',delimiter=',')
    
    data = np.zeros((marking.shape[0],cnf.square_size, cnf.square_size, 3), dtype=np.float)
    
    tlp = TileDataProvider(cnf.map_source, cnf.square_size)
    
    for x in range(0,marking.shape[0]):
        im = tlp.load_tile_data(marking[x][0],marking[x][1])
        data[x] = im
    
    tlp.destruct()
    
    x_train = data[0:cnf.training_count:1]   #обучающее множество
    x_test  = data[cnf.training_count:100:1] #проверочное множество
    
    y_train = keras.utils.to_categorical(marking[0:cnf.training_count,2], num_classes=5)
    y_test = keras.utils.to_categorical(marking[cnf.training_count:100,2], num_classes=5)
    
    model = init_model()

    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    model.fit(x_train, y_train, batch_size=20, epochs=400)
    model.save_weights('./model_weights.h5')
elif(len(sys.argv)>1 and sys.argv[1]=="load"):
    model.load_weights('./model_weights.h5')
else:
    print("usage: main.py train|load")
    quit()

#score = model.evaluate(x_test, y_test, batch_size=32)
predict_train = model.predict(x_train,verbose=1);
predict_test = model.predict(x_test,verbose=1);
print(predict_train)
print(predict_test)
print(model.predict_classes(x_train))
print(marking[0:cnf.training_count,2])
print(model.predict_classes(x_test))
print(marking[cnf.training_count:100,2])