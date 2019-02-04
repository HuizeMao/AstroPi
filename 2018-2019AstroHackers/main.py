#import necessary libraries
from sense_hat import SenseHat
from picamera import PiCamera
from logzero import logger
from math import sin,cos
from time import sleep
import datetime
import logging
import logzero
import random
import ephem
import math
import csv
import os

#define directory path of this file
dir_path = os.path.dirname(os.path.realpath(__file__))

# Connect to the Sense Hat
sh = SenseHat()

# Set a logfile name
logzero.logfile(dir_path+"/data_test.csv")

# Set a custom formatter
formatter = logging.Formatter('');
logzero.formatter(formatter)

#write header
logger.info("time,ISS_roll_x(rad),ISS_pitch_y(rad),yaw_z(rad),ISS_lattitude,ISS_longitude, humidity,temperature(C),pressure(Millibars), IssDistanceToEarthSeaLevel(km),magint_x(microteslas),magint_y(microteslas),magint_z(microteslas), ISS_to_Sun(km)")

# define some colours - keep brightness low
g = [0,50,0]
o = [0,0,0]

# define a simple image
img1 = [
    g,g,g,g,g,g,g,g,
    o,g,o,o,o,o,g,o,
    o,o,g,o,o,g,o,o,
    o,o,o,g,g,o,o,o,
    o,o,o,g,g,o,o,o,
    o,o,g,g,g,g,o,o,
    o,g,g,g,g,g,g,o,
    g,g,g,g,g,g,g,g,
]


# run a loop for 2 minutes
sh.set_pixels(img1)

#This function is used to compute the distance between the ISS and the sun
def SphericalDistance(r,theta,phi,r2,theta2,phi2):
    """inputs:
     point1 vector:
        r = distance from the point 'p1' to the origin of the coordinate system(center of the earth)
        theta = lattitude of the point p1 in relation to earth
        phi = longitude of the point p1 in relation to earth
     point2 vector:
        r2 = distance of the point 'p2' to the center of the coordinate system(center of the earth)
        theta2 = lattitude of the point p2 in relation to earth
        phi2 = longitude of the point p2 in relation to earth
    outputs:
        d = distance of these two point vectors"""
    #the way the function works is that it first turns this into Cartesian coordinates and then compute
    #https://math.stackexchange.com/questions/833002/distance-between-two-points-in-spherical-coordinates
    d = math.sqrt(r**2 + r2**2 - 2*r*r2 * (sin(theta)*sin(theta2)*cos(phi-phi2)+cos(theta)*cos(theta2)))
    return d


def active_status():
    """
    A function to update the LED matrix regularly
    to show that the experiment is progressing
    """
    # a list with all possible rotation values
    orientation = [0,90,270,180]
    # pick one at random
    rot = random.choice(orientation)
    # set the rotation
    sh.set_rotation(rot)

#create a datetime var to store the start time
start_time = datetime.datetime.now()
#create a datetime var to show the current time
now_time = datetime.datetime.now()

while (now_time < start_time + datetime.timedelta(minutes=178)):
    try:
        """
        Saves the following into a data csv file:
            recorded data (temp,humidity,pressure,orientation,magnetic intesit[using SenseHat])
            calculated data (earth distance to the sun[using spherical coordinate calculation])
        """
        ###Read data(temp,humidity,pressure,orientation,magnetic intesitt) from the Sense Hat
        ##rounded to 4 decimal places
        temperature = round(sh.get_temperature(),4)
        humidity = round(sh.get_humidity(),4)
        pressure = round(sh.get_pressure(),4)
        orientation = sh.get_orientation_radians()
        (roll_x,pitch_y,yaw_z) = (orientation['roll'], orientation['pitch'], orientation['yaw']) #orientation in three axis
        (roll_x,pitch_y,yaw_z) = (round(roll_x,4), round(pitch_y,4), round(yaw_z,4))
        raw = sh.get_compass_raw() #A dictionary with x, y and z, representing the magnetic intensity of the axis in microteslas (µT).
        (mag_x,mag_y,mag_z) = (raw['x'],raw['y'],raw['z'])
        (mag_x,mag_y,mag_z) = (round(mag_x,4),round(mag_y,4),round(mag_z,4))


        ###calculate the distance from the ISS to the sun using spherical coordinates

        ##compute the spherical co-ordinates of the sun(center of earth as origin)
        sun = ephem.Sun() #sun object
        sun.compute() #compute with current time
        #first parameter(r) of the coordinate system(distance in km)
        r = sun.earth_distance * 149598073
        #second parameter(theta) of the spherical coordinate(lattitude of the sun in relation to earth) in radians
        theta = ephem.Ecliptic(sun).lat
        #third parameter(phi) of the spherical coordinate(longitude of the sun in relation to earth) in radians
        phi = ephem.Ecliptic(sun).lon


        ##compute ISS's spherical coordinates
        #two line elements and name for creating ISS object
        name = "ISS (ZARYA)"
        line1 = "1 25544U 98067A   19034.46824074  .00002002  00000-0  38666-4 0  9999"
        line2 = "2 25544  51.6436 306.2336 0005085 347.5491 355.7714 15.53226696154479"
        iss = ephem.readtle(name, line1, line2)  #ISS object
        iss.compute() #compute with current time

        #calculate the spherical co-ordinates of the ISS(center of earth as origin)
        IssDistanceToEarthSeaLevel = iss.elevation # Geocentric height of iss above sea level (m)
        #first parameter(r) of the coordinate system(distance in km)
        r_prime = iss.elevation/1000 + 6371
        #second parameter(theta) of the spherical coordinate(lattitude of the sun in relation to earth) in radians
        theta_prime = iss.sublat
        #third parameter(phi) of the spherical coordinate(longitude of the sun in relation to earth) in radians
        phi_prime = iss.sublong

        ##compute distance from the ISS to the sun
        ISS_to_Sun = SphericalDistance(r,theta,phi,r_prime,theta_prime,phi_prime)
        ISS_to_Sun = round(ISS_to_Sun,4)
        # Save the data to the file
        logger.info("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s",now_time,roll_x,pitch_y,yaw_z, theta_prime,phi_prime, humidity,temperature,pressure, IssDistanceToEarthSeaLevel,mag_x,mag_y,mag_z, ISS_to_Sun)

        active_status()
        sleep(5)

        # update the current time
        now_time = datetime.datetime.now()
        
    except Exception as e:
        logger.error("An error occurred: " + str(e))
