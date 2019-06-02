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
our_data[['Theta','Phi','R']] = (our_data[['Theta','Phi','R']] - our_data[['Theta','Phi','R']].mean()) / (our_data[['Theta','Phi','R']].max() - our_data[['Theta','Phi','R']].min())
Theta = np.array(our_data[['Theta']])
Phi = np.array(our_data[['Phi']])
R = np.array(our_data[['R']])


###training process of the model

##design the tensorflow graph(how the function will look like)
#define the input predictor that is used to predict the positions(independent variable = time step)
x = np.arange(19354,step=2) #define x as the amount of seconds has passed since the beginning of the data collection(5:44:18 PM as beginning reference point)
x = np.reshape(x,(x.shape[0],1)) #reshape in order for model training
n = x.shape[0] #number of training example

#define targets of the predictor(dependent variable = positional data[y])
y = R #first coordinate of the spherical coordinate to predict(r = distance from the ISS to the center of Earth)
y2 = Theta #second coordinate of the spherical coordinate to predict(Theta = lattitude relative to earth)
y3 = Phi # third coordinate of the spherical coordinate to predict(Theta = longitude relative to earth)

#define predictive model with features(sinusoidal function) y = a sin (bx+c)

#tf.placeholders for the input and output of the network. Placeholders are
# variables which we need to fill in when we are ready to compute the graph. because they could vary for different dataset input
X = tf.placeholder(tf.float32,name='X_input')
Y = tf.placeholder(tf.float32,name='y_true')
n_observations = 9677 #number of training examples

#define the variable that the machine learning process is going to learn
a = tf.Variable(tf.random_normal([1]), name='amplitude') #a: affects the amplitude of the sinusoidal function
b = tf.Variable(tf.random_normal([1]), name='period') #b: affects the period of the sinusoidal function
c = tf.Variable(tf.random_normal([1]), name='shifts') #c: affects the position of the sinusoidal function
d = tf.Variable(tf.random_normal([1]), name='bias') #bias term

#define the predictive model( 'y = A sin (B(x - C)) + D' )
Y_pred =tf.add(tf.multiply(tf.math.sin(tf.multiply(tf.subtract(X,c),b)),a),d)
#Y_pred = tf.multiply(tf.math.sin(tf.add(tf.multiply(b,X),c)),a)


##initializing model
# Mean Squared Error Cost Function
cost = tf.reduce_sum(tf.pow(Y_pred-Y, 2)) / (2 * n)

#loss function would be the function that the ML algoirhm will try to optimize on
loss = tf.losses.mean_squared_error(labels=y2,predictions=Y_pred)

# %% Use gradient descent to optimize W,b
# Performs a single step in the negative gradient
learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Here we tell tensorflow that we want to initialize all
# the variables in the graph so we can use them
init = tf.global_variables_initializer()

# starting a new session, and define fulfil all the variables
sess = tf.Session()
sess.run(init)

##visualization of the computing graph(for debugging and clearity)
#tensorboard is activated through:
"""#tensorboard --logdir=./graphs"""
"""writer = tf.summary.FileWriter('./graphs',sess.graph)"""

##execute the graph and train
epoch = 1000 #number of itertions of corrections of the 'weights'(a,b,c)

for step in range(epoch):
    # Feeding each data point into the optimizer using Feed Dictionary
        for (_x, _y) in zip(x, y):
            sess.run(optimizer, feed_dict = {X : _x, Y : _y})

        # Displaying the result after every 50 epochs
        if (epoch + 1) % 1 == 0:
            # Calculating the cost a every epoch
             co = sess.run(cost, feed_dict = {X : x, Y : y})
             print("Epoch", (epoch + 1), ": cost =", co, "a =", sess.run(a), "b =", sess.run(b), "c =", sess.run(c), "d=",sess.run(d))


    #save_path = saver.save(sess, "/tmp/model.ckpt")#save the model
    #print("Model saved in path: %s" % save_path)
