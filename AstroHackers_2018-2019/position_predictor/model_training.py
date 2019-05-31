##load all the necessary library
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from keras.layers import Input
import tensorflow as tf
import pandas as pd
import numpy as np

##read the data we collected, including atmospheric and positiional data
our_data = pd.read_csv('data_analysis_preprocess/our_data.csv')

##normalize the position coordinate(y)
our_data[['Theta','Phi','R']] = (our_data[['Theta','Phi','R']] - our_data[['Theta','Phi','R']].mean())
(our_data[['Theta','Phi','R']].max() - our_data[['Theta','Phi','R']].min())
Theta = np.array(our_data[['Theta']])
Phi = np.array(our_data[['Phi']])
R = np.array(our_data[['R']])


###training process of the model

##design the tensorflow graph(how the function will look like)
#define the input predictor that is used to predict the positions(independent variable = time step)
x = np.arange(19354,step=2) #define x as the amount of seconds has passed since the beginning of the data collection(5:44:18 PM as beginning reference point)
x = np.reshape(x,(x.shape[0],1)) #reshape in order for model training

#define targets of the predictor(dependent variable = positional data[y])
y = R #first coordinate of the spherical coordinate to predict(r = distance from the ISS to the center of Earth)
y2 = Theta #second coordinate of the spherical coordinate to predict(Theta = lattitude relative to earth)
y3 = Phi # third coordinate of the spherical coordinate to predict(Theta = longitude relative to earth)

#define predictive model with features(sinusoidal function) y = a sin (bx+c)

#tf.placeholders for the input and output of the network. Placeholders are
# variables which we need to fill in when we are ready to compute the graph. because they could vary for different dataset input
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
n_observations = 9677 #number of training examples

#define the variable that the machine learning process is going to learn
a = tf.Variable(tf.random_normal([1]), name='amplitude') #a: affects the amplitude of the sinusoidal function
b = tf.Variable(tf.random_normal([1]), name='period') #b: affects the period of the sinusoidal function
c = tf.Variable(tf.random_normal([1]), name='shifts') #c: affects the position of the sinusoidal function

#define the predictive model(y_pred = a sin (bx+c))
Y_pred = tf.multiply(tf.math.sin(tf.add(tf.multiply(b,X),c)),a)

# %% Loss function will measure the distance difference between our observations
# and predictions we made and average over them.
cost = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (n_observations - 1)


# %% Use gradient descent to optimize W,b
# Performs a single step in the negative gradient
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

##execute the graph and train
# %% We create a session to use the graph
n_epochs = 100
with tf.Session() as sess:
    # Here we tell tensorflow that we want to initialize all
    # the variables in the graph so we can use them
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # Fit all training data
    prev_training_cost = 0.0

    for epoch_i in range(n_epochs):
        x = sess.run(x)
        y = sess.run(y)
        sess.run(optimizer, feed_dict={X: x, Y: y}) #run one step of gradient descent(correction of a,b,c)
        training_cost = sess.run(cost, feed_dict={Y_pred: Y_pred, Y: y}) # compute how wrong we are in this iteration
        print(training_cost)#print the cost every iteration

        # Allow the training to quit if we've reached a minimum
        if np.abs(prev_training_cost - training_cost) < 0.000001:
            break
        prev_training_cost = training_cost

    save_path = saver.save(sess, "/tmp/model.ckpt")#save the model
    print("Model saved in path: %s" % save_path)
