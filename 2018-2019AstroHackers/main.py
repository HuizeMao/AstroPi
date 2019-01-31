from sense_hat import SenseHat
from picamera import PiCamera
from logzero import logger
from time import sleep
import datetime
import logging
import logzero
import random
import ephem
import csv
import os

#define directory path of this file
dir_path = os.path.dirname(os.path.realpath(__file__))

# Connect to the Sense Hat
sh = SenseHat()

# Set a logfile name
logzero.logfile(dir_path+"/data01.csv")

# Set a custom formatter
formatter = logging.Formatter('%(name)s - %(asctime)-15s - %(levelname)s: %(message)s');
logzero.formatter(formatter)

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
        # Read data from the Sense Hat, rounded to 4 decimal places
        temperature = round(sh.get_temperature(),4)
        humidity = round(sh.get_humidity(),4)
        pressure = round(sh.get_pressure(),4)
        orientation = sh.get_orientation_radians()
        (roll_x,pitch_y,yaw_z) = (orientation['roll'], orientation['pitch'], orientation['yaw']) #orientation in three axis
        raw = sh.get_compass_raw() #A dictionary with x, y and z, representing the magnetic intensity of the axis in microteslas (µT).
        (mag_x,mag_y,mag_z) = (raw['x'],raw['y'],raw['z'])
        ##calculate the distance from the ISS to the sun
        #calculation of the distance from the sun to an observer
        sun = ephem.Sun()

        #Iss distance from the same observer
        name = "ISS (ZARYA)"
        line1 = "1 25544U 98067A   18032.92935684  .00002966  00000-0  52197-4 0  99911 25544U 98067A   18032.92935684  .00002966  00000-0  52197-4 0  9991"
        line2 = "2 25544  51.6438 332.9972 0003094  62.2964  46.0975 15.54039537 97480"

        iss = ephem.readtle(name, line1, line2)
        iss.compute()
        IssDistanceToEarthSeaLevel = iss.elevation # Geocentric height of iss above sea level (m)
        lat,lon = iss.sublat, iss.sublong #ISS position above earth

        # Save the data to the file
        logger.info("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s", roll_x,pitch_y,yaw_z, lat,lon, humidity,temperature,pressure, IssDistanceToEarthSeaLevel,mag_x,mag_y,mag_z)

        active_status()
        sleep(15)

        # update the current time
        now_time = datetime.datetime.now()
    except Exception as e:
        logger.error("An error occurred: " + str(e))
