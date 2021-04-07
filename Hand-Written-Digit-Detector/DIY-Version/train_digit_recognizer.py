#train_digit_recognizer.py

#Michael Ramirez

# Neural Network that currently has a 80% accuracy

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


batch_size = 128
num_classes = 10
epochs = 15

model = Sequential([Flatten(input_shape = input_shape), 
                    Dense(420, activation = 'relu'),
                    Dense(10, activation = 'softmax')])


model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))


score = model.evaluate(x_test, y_test, verbose=0)
print('loss: ', score[0])
print('Accuracy: ', score[1])

model.save('mnist.h5')
print("model saved as mnist.h5")

