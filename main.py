#use pip install numpy before you do anything, put on terminal
#use pip install opencv-python
#use pip install matplotlib
#use pip install tensorflow
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utilis.normalize(x_train, axis=1)
x_test = tf.keras.utilis.normalize(x_test, axis=1)

model= tf.keras.model.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorial_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

model.save('handwritten.model')

#Now run to test the training data
#Once finished comment the code above and now run the Neeural Network system

model = tf.keras.models.load_model('handwritten.model')

#Uncomment to test the data
#mnist = tf.keras.datasets.mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

#x_train = tf.keras.utilis.normalize(x_train, axis=1)
#x_test = tf.keras.utilis.normalize(x_test, axis=1)

loss,accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)

#If you use ms_paint make the brushes pixels and uncheck maintain aspect ratio and
# then change horizontal and vertical length to 28
#Then save numbers to a directory
#After finishing the steps stated before remove the test evaluation part (line 33-43)
#Then comment out lines 11-15

#Uncomment code below once finished

#image_number=1
#while os.path.isfile(f"digits/digit{image_number}.png"):
#    try:
#        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
#        img = np.invert(np.array([img]))
#        prediction = model.predict(img)
#        print(f"This number is most likely {np.argmax(prediction)}")
#        plt.imshow(img[0], cmap=plt.cm.binary)
#        plt.show()
#    except:
#        print("Error")
#    finally:
#        image_number += 1