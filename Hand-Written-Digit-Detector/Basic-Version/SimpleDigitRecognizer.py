#SimpleDigitRecognizer.py

#Michael Ramirez

# This is a simple way of training and predicting handwritten digits using Tensorflow.


import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt 

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

x_train = x_train 
y_train = y_train 

model = keras.Sequential([keras.layers.Flatten(input_shape = (28,28)), 
                          keras.layers.Dense(226, activation = 'relu'),
                          keras.layers.Dense(10, activation = 'softmax')])


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 8)

test_loss, test_accuracy = model.evaluate(x_train, y_train)

print("\nLoss: ", test_loss)
print('--------------------------')
print('Accuracy: ', test_accuracy)

predictions = model.predict(x_test)


for i in range(10):
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " +  str(y_test[i]))
    plt.title("Prediction: "+ digits[np.argmax(predictions[i])])
    plt.show()

