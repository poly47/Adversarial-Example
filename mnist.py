from keras.datasets import mnist
## preprocess the dataset
# load mnist dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# convert features to float type
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalization
X_train /= 255
X_test /= 255
# flatten
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
# convert label to one-hot type
from keras.utils import np_utils
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# construct the model
from keras.models import Sequential
from keras.layers import Dense, Activation
model = Sequential()
model.add(Dense(128, input_shape = (784, )))
model.add(Activation('relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile the model
model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics=['accuracy'])

# fit model
model.fit(X_train, Y_train, batch_size = 32, epochs = 20, verbose = 0)

# save model
model.save('my_model.h5')

# evaluate model
print("The accuracy is:")

# save the model
print(model.evaluate(X_test, Y_test, verbose=0))


# calculate hessian
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=(1, 784))
#predict = model(x)
#x_f = tf.reshape(x, [784])
x_f = x[0]
x_i = tf.expand_dims(x_f,0)
predict = model(x_i)
y = tf.placeholder(tf.float32)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = predict))
hess = tf.hessians(cost, x_f)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    hessian_matrix = sess.run(hess, feed_dict={x:X_test[1:2], y:Y_test[1]})

# find sorted eigenvalue
import numpy as np
from numpy import linalg as la
w, v = la.eig(hessian_matrix)
sorted_eigenvalue = sorted(np.absolute(w))


print(sorted_eigenvalue)


   
