##load all the necessary library
import matplotlib.pyplot as plt
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


##configure model
# Mean Squared Error Cost Function
cost = loss = tf.reduce_sum(tf.pow(Y_pred - Y,2))/(2*m)

# %% Use gradient descent to optimize W,b
# Performs a single step in the negative gradient
learning_rate = tf.placeholder(tf.float32, shape=None, name='learning_rate') #the learning rate of the optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.minimize(cost)
grads_and_vars = optimizer.compute_gradients(cost) #compute the gradient for the sake of debug

###debuge part
# Name scope allows you to group various summaries together
# Summaries having the same name_scope will be displayed on the same row
with tf.name_scope('performance'):
    # Summaries need to be displayed
    # Whenever you need to record the loss, feed the mean loss to this placeholder
    tf_loss_ph = tf.placeholder(tf.float32,shape=None,name='loss_summary')
    # Create a scalar summary object for the loss so it can be displayed
    tf_loss_summary = tf.summary.scalar('loss', tf_loss_ph)

    # Whenever you need to record the loss, feed the mean test accuracy to this placeholder
    tf_accuracy_ph = tf.placeholder(tf.float32,shape=None, name='accuracy_summary')
    # Create a scalar summary object for the accuracy so it can be displayed
    tf_accuracy_summary = tf.summary.scalar('accuracy', tf_accuracy_ph)

# Gradient norm summary
for g,v in grads_and_vars:
    with tf.name_scope('gradients'):
        tf_last_grad_norm = tf.sqrt(tf.reduce_mean(g**2))
        tf_gradnorm_summary = tf.summary.scalar('grad_norm', tf_last_grad_norm)
        break

# Merge all summaries together
performance_summaries = tf.summary.merge([tf_loss_summary,tf_accuracy_summary])

###train
# Here we tell tensorflow that we want to initialize all
# the variables in the graph so we can use them
init = tf.global_variables_initializer()

#define saver object to save the model
saver = tf.train.Saver()

# starting a new session, and define fulfil all the variables
sess = tf.Session()
sess.run(init)

##execute the graph and train (uses mini-batch, which means it trains a sample[100] of th entire data in one iteration and eventually will cover every sub-sample)
epoch = 10000 #number of itertions of corrections of the 'weights'(a,b,c)
batch_size = 9677 #sample size
n_batches = int(np.ceil(m / batch_size)) #number of samples
co_list = list() #define cost array that keeps track of how the error rate changes[later used to be plotted]

#this is a function that will be later used to fetch a batch(sample)
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    # Set random indices
    indices = np.random.randint(m, size=batch_size) #this variable has size [100,1] with 100 random values from range 1:data size
    # Define a batch X based on previous indices
    X_batch = x[indices]
    # y batch
    y_batch = y2[indices-2] #the result shows that the batch doesn't match up, and the index are 2 different from each other, therefore the group decided just minus two here to fix the error
    return X_batch, y_batch

saver.restore(sess,"models/theta_predictor.ckpt") #restore variables

for step in range(epoch):
    # Feeding each a batch into the optimizer using Feed Dictionary
    for batch_index in range(n_batches):# for each batch index
        X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)#get the mini-batch used to train
        sess.run(train_step, feed_dict={X: X_batch, Y: y_batch, learning_rate: 0.01}) #run gradient descent to correct weights

    #calulate the step cost and add it into an array that records the progress of training
    step_co = float(sess.run(cost, feed_dict = {X : x, Y : y2}))
    co_list.append(step_co)

    # Displaying the result after every 50 epochs
    if (step + 1) % 50 == 0:
        # Calculating the cost a every epoch
        co = sess.run(cost, feed_dict = {X : x, Y : y2})
        print("Epoch", (step + 1), ": cost =", co, "a =", sess.run(a), "b =", sess.run(b), "c =", sess.run(c), "d=",sess.run(d))

print("final cost after this session of training: {}".format(sess.run(cost, feed_dict = {X : x, Y : y2})))
co_list = np.round(np.array(co_list),2) # round the cost to two decimal places

#print(sess.run(Y_pred,feed_dict={X:x}))
"""#compare predictions Vs y_true
print(sess.run(Y_pred,feed_dict={X:x}))
print(y2)"""

#plot the training history and their relative cost change
plt.plot(co_list)
plt.title('Training cost Vs Iterations')
plt.ylabel('training cost')
plt.xlabel('epochs')
plt.show()
##print(sess.run(Y_pred, feed_dict = {X : x, Y : y2}))

##visualization of the computing graph(for debugging and clearity)
#tensorboard is activated through:
"""#tensorboard --logdir=./graphs"""
writer = tf.summary.FileWriter('./tensorboard_debug',sess.graph)
#tensorboard --logdir= ./graphs --debugger_port 7000


###save path
##save_path = saver.save(sess, "model/r_predictor.ckpt")#save the model(that predicts r)
save_path = saver.save(sess, "models/theta_predictor.ckpt")#save the model(that predicts theta)
##save_path = saver.save(sess, "model/phi_predictor.ckpt")#save the model(that predicts phi)
