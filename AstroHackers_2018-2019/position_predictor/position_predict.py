"""This file predicts the following position co-ordinates:
y = R #first coordinate of the spherical coordinate to predict(r = distance from the ISS to the center of Earth)
y2 = Theta #second coordinate of the spherical coordinate to predict(Theta = lattitude relative to earth)
y3 = Phi # third coordinate of the spherical coordinate to predict(Theta = longitude relative to earth)"""
##load all the necessary library
from sense_hat import SenseHat
import tensorflow as tf
import datetime as dt
import pandas as pd
import numpy as np

# Connect to the Sense Hat
sh = SenseHat()

###tensorflow part that restores the model
##define tensorflow saver object(used to restore trained model)
saver = tf.train.Saver()
#initialize variables for the model graph in order for it to be restored later
X = tf.placeholder(tf.float32,name='X_input')
##set up the inital variables needed to predict the position, and would be later be replaced by the trained ones
a = tf.Variable(tf.zeros([1]), name='amplitude') #a: affects the amplitude of the sinusoidal function
b = tf.Variable(tf.zeros([1]), name='period') #b: affects the period of the sinusoidal function
c = tf.Variable(tf.zeros([1]), name='shifts') #c: affects the position of the sinusoidal function
d = tf.Variable(tf.zeros([1]), name='bias') #bias term

#define prediction
Y_pred =tf.add(tf.multiply(tf.math.sin(tf.multiply(tf.subtract(X,c),b)),a),d)


##while true(until manually break loop, make a predictions about the current position)
while True:
    try:
        reference_time = dt.datetime(2019,4,4,17,44,18) #first time point that we trained our models on(x is the number of seconds that passed this reference time)
        now_time = dt.datetime.now() #get the time now
        x = (now_time-reference_time).total_seconds() #input to the model-number of seconds passed since the reference time
        ##make a prediction using the tensorflow graph
        with tf.Session() as sess: #run a session
            #predict R
            saver.restore(sess,"models/r_predictor.ckpt")
            predicted_R = sess.run(Y_pred, feed_dict = {X:x})

            #predict Theta
            tf.reset_default_graph() #reset graph and import a model that predicts theta
            saver.restore(sess,"models/theta_predictor.ckpt") #restore variables
            predicted_theta = sess.run(Y_pred, feed_dict = {X:x})

            #predict Phi
            saver.restore(sess,"models/phi_predictor.ckpt")
            predicted_R = sess.run(Y_pred, feed_dict = {X:x})
            tf.reset_default_graph() #reset graph and import a model that predicts phi

            print("The predicted position of ISS(Zarya) in spherical co-ordinate is at: ({},{},{}) [Distance to center of Earth, Lattitude,Longitude]".format(predicted_r,predicted_theta,predicted_phi))

        sleep(2)
