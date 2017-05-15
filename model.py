from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization

def init_model():
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
    
    return model