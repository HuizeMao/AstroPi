import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
##
#y = (distance_to_an_origin, latitude, longitude)

"""pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)"""

#http://www.isstracker.com/historical site for look up ISS positions
#https://projects.raspberrypi.org/en/projects/astro-pi-flight-data-analysis/8 ESA site

cwd = os.getcwd()

our_data = pd.read_csv('our_data.csv')

columbus_data = pd.read_csv('columbus_data.csv')
#pd.Dataframe.replace(our_data[["Theta"]],Theta,regex=False)
#print(our_data[["Theta"]].head(100))

print(columbus_data)




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
