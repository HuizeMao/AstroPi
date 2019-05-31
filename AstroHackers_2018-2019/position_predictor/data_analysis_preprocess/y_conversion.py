import pandas as pd
import numpy as np
import os

our_data = pd.read_csv('theta_phi.csv',delimiter=",|:")

t_degree = np.array(our_data['t_degree'].values.tolist(),dtype = np.int16)
t_minute = np.array(our_data['t_minute'].values.tolist(),dtype = np.int16)
t_second = np.array(our_data['t_second'].values.tolist(),dtype = np.int16)

p_degree = np.array(our_data['p_degree'].values.tolist(),dtype = np.int16)
p_minute = np.array(our_data['p_minute'].values.tolist(),dtype = np.int16)
p_second = np.array(our_data['p_second'].values.tolist(),dtype = np.int16)

def convert_to_y(degree,minute,second):
    output = np.empty(degree.shape)
    counter = 0
    for d in degree:
        if d<=0:
            output[counter] = (degree[counter] - minute[counter] - second[counter])
        elif d >0:
            output[counter] = (degree[counter] + minute[counter] + second[counter])
        counter+=1
    return output

t_MinuteToDegree = t_minute/60
t_SecondToDegree = (t_second/60)/60
p_MinuteToDegree = p_minute/60
p_SecondToDegree = (p_second/60)/60

theta = convert_to_y(t_degree,t_MinuteToDegree,t_SecondToDegree)
phi = convert_to_y(p_degree,p_MinuteToDegree,p_SecondToDegree)

theta = np.round(theta,3)
theta = theta.reshape(9677,1)
phi = np.round(phi,3)
phi = phi.reshape(9677,1)

y = np.concatenate((theta,phi), axis = 1)
print(y[0:10])
print(y.shape)
np.savetxt("y.csv", y, delimiter=",",header = "Theta", fmt='%1.3f')

#print(t_degree[0],t_MinuteToDegree[0],t_SecondToDegree[0])



"""
print(t_degree[0])
print(t_minute[0])
print(t_second[0])

print(p_degree[0])
print(p_minute[0])
print(p_second[0])
"""

"""theta = np.char.split(theta, sep = ':')
phi = np.char.split(phi,sep = ':')"""
