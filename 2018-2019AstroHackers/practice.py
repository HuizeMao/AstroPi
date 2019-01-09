from sense_hat import SenseHat
from random import uniform

sense = SenseHat()
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0,0,255)
text_rgb = (123,24,255)
#num = uniform(0,10)

#detects air pressure
pressure = sense.get_pressure()
pressure = round(pressure,1)
print(pressure)
# detects temperature
temp = sense.get_temperature()
temp = round(temp,1)
print(temp)
#detects humidity
humidity = sense.get_humidity()
humidity = round(humidity,1)
print(humidity)

"""
According to online documentation, the International Space Station maintains these conditions at the following levels:

Temperature: 18.3-26.7 Celsius

Pressure: 979-1027 millibars

Humidity: around 60%
"""

if temp > 18.3 and temp < 26.7:
    bg = green
else:
    bg = red
#gets the pitch, row, yaw of the orientation
o = sense.get_orientation()
pitch = o["pitch"]
roll = o["roll"]
yaw = o["yaw"]

print("pitch {0} roll {1} yaw {2}".format(pitch, roll, yaw))

#detects the amount of G force acting on each axis{x,y,z}
#If any axis has ±1G, then you know that axis is pointing downwards.
acceleration = sense.get_accelerometer_raw()
x = acceleration['x']
y = acceleration['y']
z = acceleration['z']

x=round(x, 0)
y=round(y, 0)
z=round(z, 0)

print("x={0}, y={1}, z={2}".format(x, y, z))
#reset where the text orientation according to where resberry pi is facing
if x == -1:
    sense.set_rotation(180)
elif y == 1:
    sense.set_rotation(90)
elif y == -1:
    sense.set_rotation(270)
else:
    sense.set_rotation(0)

x = abs(x)
y = abs(y)
z = abs(z)

if x > 1 or y > 1 or z > 1:
    sense.show_letter("!", red)

sense.show_message("Hello World",0.1, text_colour = text_rgb, back_colour=bg)
sense.clear()
i = 0
while i < 3:
    for event in sense.stick.get_events():
        print(event.direction, event.action)
        i += 1







