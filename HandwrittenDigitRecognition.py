import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist #we already know the flassifications of all these digits
(x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz") #the load data function loads the mnist data and already splits it into training and testing data.


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#create a new model
model = tf.keras.models.Sequential() #an ordinary feedforward neural network

#build the model

#add some layers
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) #flatten the layer with all the pixels off each individual image of a handwritten digit and we feed that into the input layer and then this input layer is followed by a dense layer
#add the dense layer
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu)) #number of neurons that we are gonna have in this layer. The activation function is a simple rectify linear unit function
#add the second hidden layer
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
#output layer
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)) #the activation function is the function that tries to take all the outputs

#we need to comple them all
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#fit the model
model.fit(x_train, y_train, epochs=5) #the model is going to repeat the whole process 3 times

#evaluate them all
loss, accuracy = model.evaluate(x_test, y_test)

print(accuracy)
print(loss)

model.save('digits.model')

#we are creating a for loop to import all our images
for x in range(1,6):
    #image read funciton
    img = cv.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    #how likely is it to be a certain number, we want to get only the clasifications
    print(f'The result is probably: {np.argmax(prediction)}')#argmax gives us the index of the max value
    #to see how it looks like
    plt.imshow(img[0],cmap=plt.cm.binary)
    plt.show()


