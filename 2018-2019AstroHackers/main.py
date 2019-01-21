import logging
import logzero
from logzero import logger
from sense_hat import SenseHat
import ephem
from picamera import PiCamera
import datetime
from time import sleep
import random
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

# Set up camera
cam = PiCamera()
cam.resolution = (1296,972)

# run a loop for 2 minutes
photo_counter = 1
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
    
while (now_time < start_time + datetime.timedelta(minutes=178)):
    try:
        # Read some data from the Sense Hat, rounded to 4 decimal places
        temperature = round(sh.get_temperature(),4)
        humidity = round(sh.get_humidity(),4)
        pressure = round(sh.get_pressure(),4)

        # get latitude and longitude
        lat, lon = get_latlon()
        
        # Save the data to the file
        logger.info("%s,%s,%s,%s,%s,%s", photo_counter,humidity, temperature, pressure, lat, lon )
        
        # use zfill to pad the integer value used in filename to 3 digits (e.g. 001, 002...)
        cam.capture(dir_path+"/photo_"+ str(photo_counter).zfill(3)+".jpg")
        photo_counter+=1
        active_status()
        sleep(15)
        active_status()
        sleep(15)
        
        # update the current time
        now_time = datetime.datetime.now()
    except Exception as e:
        logger.error("An error occurred: " + str(e))
