##load all the necessary library
import tensorflow as tf
import pandas as pd
import numpy as np

##read the data we collected, including atmospheric and positiional data
our_data = pd.read_csv('data_analysis_preprocess/our_data.csv')

##extract the response variables: position coordinate(y)
Theta = np.array(our_data[['Theta']])
Phi = np.array(our_data[['Phi']])
R = np.array(our_data[['R']])

###training process of the model

##design the tensorflow graph(how the function will look like)
#define the input predictor that is used to predict the positions(independent variable = time step)
x = np.arange(19354,step=2) #define x as the amount of seconds has passed since the beginning of the data collection(5:44:18 PM as beginning reference point)
x = np.reshape(x,(x.shape[0],1)) #reshape in order for model training
m = x.shape[0] #number of training example

#define targets of the predictor(dependent variable = positional data[y])
y = R #first coordinate of the spherical coordinate to predict(r = distance from the ISS to the center of Earth)
y2 = Theta #second coordinate of the spherical coordinate to predict(Theta = lattitude relative to earth)
y3 = Phi # third coordinate of the spherical coordinate to predict(Theta = longitude relative to earth)

print(np.concatenate((x,y2),axis=1))

#define predictive model with features(sinusoidal function) y = a sin (bx+c)

#tf.placeholders for the input and output of the network. Placeholders are
# variables which we need to fill in when we are ready to compute the graph. because they could vary for different dataset input
X = tf.placeholder(tf.float32,name='X_input')
Y = tf.placeholder(tf.float32,name='y_true')

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
cost = tf.reduce_sum(tf.pow(Y_pred-Y, 2)) / (2 * m)

# %% Use gradient descent to optimize W,b
# Performs a single step in the negative gradient
learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Here we tell tensorflow that we want to initialize all
# the variables in the graph so we can use them
init = tf.global_variables_initializer()

#define saver object to save the model
saver = tf.train.Saver()

# starting a new session, and define fulfil all the variables
sess = tf.Session()
sess.run(init)

##visualization of the computing graph(for debugging and clearity)
#tensorboard is activated through:
"""#tensorboard --logdir=./graphs"""
"""writer = tf.summary.FileWriter('./graphs',sess.graph)"""

##execute the graph and train (uses mini-batch, which means it trains a sample[100] of th entire data in one iteration and eventually will cover every sub-sample)
epoch = 3 #number of itertions of corrections of the 'weights'(a,b,c)
batch_size = 100 #sample size
n_batches = int(np.ceil(m / batch_size)) #number of samples

#this is a function that will be later used to fetch a batch(sample)
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    # Set random indices
    indices = np.random.randint(m+1, size=batch_size) #this variable has size [100,1] with 100 random values from range 1:data size
    print(indices)
    # Define a batch X based on previous indices
    X_batch = x[indices]
    # y batch
    y_batch = y[indices-2] #the result shows that the batch doesn't match up, and the index are 2 different from each other, therefore the group decided just minus two here to fix the error
    return X_batch, y_batch

saver.restore(sess,"models/theta_predictor.ckpt") #restore variables

for step in range(epoch):
    # Feeding each a batch into the optimizer using Feed Dictionary
    for batch_index in range(n_batches):# for each batch index
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)#get the mini-batch used to train
            sess.run(optimizer, feed_dict={X: X_batch, y: y_batch}) #run gradient descent to correct weights

    print("{} iteration completed".format(step))#display progress

    # Displaying the result after every 50 epochs
    if (epoch + 1) % 50 == 0:
        # Calculating the cost a every epoch
        co = sess.run(cost, feed_dict = {X : x, Y : y2})
        print("Epoch", (epoch + 1), ": cost =", co, "a =", sess.run(a), "b =", sess.run(b), "c =", sess.run(c), "d=",sess.run(d))

##print(sess.run(Y_pred, feed_dict = {X : x, Y : y2}))

###save path
##save_path = saver.save(sess, "model/r_predictor.ckpt")#save the model(that predicts r)
save_path = saver.save(sess, "models/theta_predictor.ckpt")#save the model(that predicts theta)
##save_path = saver.save(sess, "model/phi_predictor.ckpt")#save the model(that predicts phi)
