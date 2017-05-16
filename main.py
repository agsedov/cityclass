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

markup = genfromtxt("marking", dtype='int',delimiter=',')

def split_markup(markup):
    np.random.seed(50)
    classes = [0,0,0,0,0]
    for c in range(0,5):
        classes[c] = markup[markup[:,2]==c]

    train = np.concatenate( 
        (classes[0][0:30,:],
        classes[1][0:30,:],
        classes[2][0:30,:],
        classes[3][0:30,:],
        classes[4][0:30,:]), axis = 0
    );

    test = np.concatenate( 
        (classes[0][30:40,:],
        classes[1][30:40,:],
        classes[2][30:40,:],
        classes[3][30:40,:],
        classes[4][30:40,:]), axis = 0
    );
    np.random.shuffle(train)
    np.random.shuffle(test)
    return np.concatenate((train, test),axis = 0)

markup = split_markup(markup)

data = np.zeros((markup.shape[0],cnf.square_size, cnf.square_size, 3), dtype=np.float)

tlp = TileDataProvider(cnf.map_source, cnf.square_size)

for x in range(0,markup.shape[0]):
    im = tlp.load_tile_data(markup[x][0],markup[x][1])
    data[x] = im

tlp.destruct()

x_train = data[0:cnf.training_count:1]   #обучающее множество
x_test  = data[cnf.training_count:200:1] #проверочное множество

y_train = keras.utils.to_categorical(markup[0:cnf.training_count,2], num_classes=5)
y_test = keras.utils.to_categorical(markup[cnf.training_count:200,2], num_classes=5)

if(len(sys.argv)>1 and sys.argv[1]=="train"):
    model = init_model()

    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    history = model.fit(x_train, y_train, batch_size=20, epochs=300,
        validation_data=(x_test, y_test))
    print(history.history.keys())
    model.save_weights('./model_weights.h5')
    plt.figure(1) 
    plt.subplot(212)

    plt.plot(history.history['loss'])  
    plt.plot(history.history['val_loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')  
    plt.show() 
    
elif(len(sys.argv)>1 and sys.argv[1]=="load"):
    model = init_model()
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
print(markup[0:cnf.training_count,2])
print(model.predict_classes(x_test))
print(markup[cnf.training_count:200,2])