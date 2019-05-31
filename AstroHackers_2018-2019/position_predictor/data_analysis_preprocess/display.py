import matplotlib.pyplot as plt
from math import degrees
import pandas as pd
import numpy as np
import ephem
import os

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

test = pd.read_csv('two_data.csv')
print(test.head())
"""print(test[["R"]])
print(test[["Theta"]])
print(test[["Phi"]])
"""
