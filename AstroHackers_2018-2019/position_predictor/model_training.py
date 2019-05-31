##load all the necessary library
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

##read the data we collected, including atmospheric and positiional data
our_data = pd.read_csv('our_data.csv')

##normalize the position coordinate(y)
our_data[['Theta','Phi','R']] = (our_data[['Theta','Phi','R']] - our_data[['Theta','Phi','R']].mean()) /(our_data[['Theta','Phi','R']].max() - our_data[['Theta','Phi','R']].min())
Theta = np.array(our_data[['Theta']])
Phi = np.array(our_data[['Phi']])
R = np.array(our_data[['R']])


###training process of the model

##define the input predictor that is used to predict the positions(independent variable = time step)
x = np.arange(19354,step=2) #define x as the amount of seconds has passed since the beginning of the data collection(5:44:18 PM as beginning reference point)
x = np.reshape(x,(x.shape[0],1)) #reshape in order for model training

##define objective of the predictor(dependent variable = positional data[y])
y = R #first coordinate of the spherical coordinate to predict(r = distance from the ISS to the center of Earth)
y2 = Theta #second coordinate of the spherical coordinate to predict(Theta = lattitude relative to earth)
y3 = Phi # third coordinate of the spherical coordinate to predict(Theta = longitude relative to earth)

##define the model with features(sinusoidal function), and fit the model into the datapoint
model = LinearRegression() # define the predictive model
model.fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
"""print(Theta[0])
print(x.shape)"""
