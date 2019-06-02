##load all the necessary library
import tensorflow as tf
import pandas as pd
import numpy as np


#read the data we collected, including atmospheric and positiional data
our_data = pd.read_csv('data_analysis_preprocess/our_data.csv')
##normalize the position coordinate(y)
our_data[['Theta','Phi','R']] = (our_data[['Theta','Phi','R']] - our_data[['Theta','Phi','R']].mean()) / (our_data[['Theta','Phi','R']].max() - our_data[['Theta','Phi','R']].min())
Theta = np.array(our_data[['Theta']])
Phi = np.array(our_data[['Phi']])
R = np.array(our_data[['R']])
x = np.arange(19354,step=2) #define x as the amount of seconds has passed since the beginning of the data collection(5:44:18 PM as beginning reference point)
x = np.reshape(x,(x.shape[0],1)) #reshape in order for model training
n = x.shape[0] #number of training example
#define targets of the predictor(dependent variable = positional data[y])
y = R #first coordinate of the spherical coordinate to predict(r = distance from the ISS to the center of Earth)
y2 = Theta #second coordinate of the spherical coordinate to predict(Theta = lattitude relative to earth)
y3 = Phi # third coordinate of the spherical coordinate to predict(Theta = longitude relative to earth)
X = tf.placeholder(tf.float32,name='X_input')
Y = tf.placeholder(tf.float32,name='y_true')
##set up the inital variables needed to predict the position, and would be later be replaced by the trained ones
a = tf.Variable(tf.zeros([1]), name='amplitude') #a: affects the amplitude of the sinusoidal function
b = tf.Variable(tf.zeros([1]), name='period') #b: affects the period of the sinusoidal function
c = tf.Variable(tf.zeros([1]), name='shifts') #c: affects the position of the sinusoidal function
d = tf.Variable(tf.zeros([1]), name='bias') #bias term
#define prediction
Y_pred =tf.add(tf.multiply(tf.math.sin(tf.multiply(tf.subtract(X,c),b)),a),d)


for (_x, _y) in zip(x, y2):
    print(_x.shape)
    print(_y.shape)
    break

##define saver object(used to restore)
saver = tf.train.Saver()
print(y2[:100])
with tf.Session() as sess: #run a session
    saver.restore(sess,"models/theta_predictor.ckpt") #restore variables
    print(sess.run(Y_pred[:100], feed_dict = {X : x, Y : y2}))
