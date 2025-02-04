from sklearn import preprocessing
import matplotlib.pyplot as plt
from math import degrees
import pandas as pd
import numpy as np
import ephem
import os
##
#y = (distance_to_an_origin, latitude, longitude)

"""pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)"""
x = np.arange(19354,step=2) #define x as the amount of seconds has passed since the beginning of the data collection(5:44:18 PM as beginning reference point)
x = np.reshape(x,(x.shape[0],1)) #reshape in order for model training

#http://www.isstracker.com/historical site for look up ISS positions
#https://projects.raspberrypi.org/en/projects/astro-pi-flight-data-analysis/8 ESA site

cwd = os.getcwd()

def graph_2d(dataframe,x,*args):
    # gca stands for 'get current axis'
    ax = plt.gca()
    for i in args:
        dataframe.plot(kind='line',x=x,y=i,ax=ax,title='R(distance)&Theta(Latitude)&Phi(Longitude) Vs Time')
    plt.show()

our_data = pd.read_csv('our_data.csv')
our_data.Time = pd.to_datetime(our_data.Time).dt.time
columbus_data = pd.read_csv('columbus_data.csv')
columbus_data.Time = pd.to_datetime(columbus_data.Time).dt.time

columbus_data[['Theta','Phi','R']] = (our_data[['Theta','Phi','R']] - our_data[['Theta','Phi','R']].mean()) /(our_data[['Theta','Phi','R']].max() - our_data[['Theta','Phi','R']].min())
#df_norm = (our_data - our_data.mean()) / (our_data.max() - our_data.min())
##graph_2d(our_data,'Theta','Phi')
graph_2d(our_data,'x','Theta')
#graph_2d(our_data,"Time","Roll (x)","Pitch (y)","Yaw (z)")

"""print(our_data[["Phi"]].max())
print(columbus_data[["Phi"]].max())"""

#our_data.to_csv("our_data.csv",index=False)

#graph_2d(our_data,"R")


plt.show()
"""our_data.plot(kind='line',x="Time",y="R",ax =ax)
plt.show()
"""
"""one_data = pd.read_csv('Columbus_one_processed.csv')
two_data = pd.read_csv('Columbus_two_processed.csv')
columbus_data = pd.concat([one_data, two_data], axis=0)
columbus_data.to_csv("columbus_data.csv",index=False)"""

"""time = two_data[["time_stamp"]]
name = "ISS (ZARYA)"
line1 = "1 25544U 98067A   16083.59985190  .00004971  00000-0  81963-4 0  9996"
line2="2 25544  51.6431 124.8898 0002010 350.6456 158.6258 15.54232983991662"
iss = ephem.readtle(name,line1, line2)

y = np.empty((257116,3))

for index,row in time.iterrows():
    now_time = row["time_stamp"]
    iss.compute(now_time)
    r = round(iss.elevation/1000 + 6371,2)
    theta = round(degrees(iss.sublat),3)
    phi = round(degrees(iss.sublong),3)
    y[index,0] = r
    y[index,1] = theta
    y[index,2] = phi


two_data['R'] = y[:,0]
two_data['Theta'] = y[:,1]
two_data['Phi'] = y[:,2]

two_data.to_csv("two_data.csv",index=False)"""



"""our_headers = [
'Time','Roll(x)','Pitch(y)','Yaw (z)','R','theta','Phi',
'Humidity (%)','Temperature (C)','Pressure (Millibars)','Mag X (microteslas)',
'Mag Y (microteslas)','Mag Z (microteslas)','Earth Distance (km)','Sun Distance (km)'
]"""

"""#add (r,theta,phi)
IssDistanceToEarthSeaLevel = np.array(our_data['Earth Distance (km)'].values.tolist())
IssDistanceToEarthSeaLevel = IssDistanceToEarthSeaLevel.astype(np.float)
print(IssDistanceToEarthSeaLevel[0:10])
r_prime = IssDistanceToEarthSeaLevel/1000 + 6371
print(r_prime[0:10])
our_data.insert(5,"R",r_prime)
print(our_data.head())
print(our_data[["R","Theta","Phi"]].head())
#print(our_data[['Latitude','Longitude','Earth Distance (km)']].head())
#r_prime = our_data[['Earth Distance (km)']]/1000 + 6371 #first factor of coordinate
"""
"""y = pd.read_csv('y.csv')
Theta = round(y[["Theta"]],3)
Phi = round(y[['Phi']],3)
our_data[["Theta"]] = Theta
our_data[["Phi"]] = Phi"""

"""
#pd.Dataframe.replace(our_data[["Theta"]],Theta,regex=False)
#print(our_data[["Theta"]].head(100))
"""
