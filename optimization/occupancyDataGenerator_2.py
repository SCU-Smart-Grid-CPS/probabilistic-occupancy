# adapted from generateOccupancyData.ipynb

#INCLUDING SCIENTIFIC AND NUMERICAL COMPUTING LIBRARIES
#Run this code to make sure that you have all the libraries at one go.

import os
#!pip install ipypublish
#from ipypublish import nb_setup
import pandas as pd
import numpy as np
import datetime
#import matplotlib.pyplot as plt
from scipy.stats import norm
import random
from cvxopt import matrix, solvers
import csv

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# SETTINGS:
ndays = 2
startDay = "1/1/2020"
endDay = "1/3/2020"

# get hourly occupancy probability data
occ_prob = pd.read_csv('occupancy_1hr.csv',usecols=['Probability'])
# extract only the relevant days
probability_days = np.array(occ_prob.iloc[0:ndays*24])
#print (probability_days)

# Generate a random number for each hour in ndays
randNums_days = np.array(np.random.rand(ndays*24))

# Make a dataframe with dates/times as index 
columnNames =["Probability","Random Number","Dates/Times"]
dates_1hr = pd.date_range(start=startDay, end=endDay, freq="1H")
#occupancyDataFrame = pd.DataFrame([probability_days,randNums_days,dates_1hr]).T
occupancyDataFrame = pd.DataFrame([probability_days,randNums_days,dates_1hr],columnNames).T
#occupancyDataFrame.columns = columnNames
occupancyDataFrame.set_index('Dates/Times', inplace=True)

# Make another column that determines occupancy based on probability and random number
occupancyDataFrame["Occupancy"] = occupancyDataFrame['Probability']-occupancyDataFrame['Random Number']

# Currently, all of these lines will return ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
#occupancyDataFrame["Occupancy"] = np.where(occupancyDataFrame["Occupancy"]>=[0], 1, 0)
occupancyDataFrame["Occupancy"] = np.where(occupancyDataFrame["Occupancy"].ge(0), 1, 0)
#occupancyDataFrame["Occupancy"].loc[occupancyDataFrame["Occupancy"]<0] = 0
#occupancyDataFrame["Occupancy"].loc[occupancyDataFrame["Occupancy"]>0] = 1

#print(occupancyDataFrame['Occupancy'].to_numpy())


# plt.plot(range(72),occupancyDataFrame['Probability'][0:72], range(72), occupancyDataFrame['Random Number'][0:72])
#ylabel('Time [hr]')
#legend(['Probability','Random_Number'])
#plt.bar(range(72),occupancyDataFrame['Occupancy'][0:72])
#occupancyDataFrame.head()



# Calculate comfort band
sigma = 3.937 # This was calculated based on adaptive comfort being normally distributed
comfort_band_function = lambda x: (1-x)/2
occupancyDataFrame['Comfort Range'] = occupancyDataFrame['Probability'].apply(comfort_band_function)
occupancyDataFrame['Comfort Range'] = occupancyDataFrame['Comfort Range']+1/2
occupancyDataFrame['Comfort Range'] = occupancyDataFrame['Comfort Range'].apply(lambda x: norm.ppf(x)*sigma)
# display first 25 lines
occupancyDataFrame.head(25)

#Brian testing - don't use this part
# Calculate comfort band
#sigma = 3.937 # This was calculated based on adaptive comfort being normally distributed
#comfort_band_function = lambda x: (1-x)/2
#occ_comfort_range = occ_prob.apply(comfort_band_fxn)+1/2
#occ_comfort_range = occ_comfort_range.apply(lambda y: norm.ppf(y)*sigma)

# Make dataframes with 5 min timesteps... or however many you want... pad() fill empty rows.. can also do .mean() and others()
#occupancyDataFrame_5mins = occupancyDataFrame.resample('5min').pad()
#occupancyDataFrame_5mins.head()


# Make dataframes with 1 min timesteps... or however many you want... pad() fill empty rows.. can also do .mean() and others()
#occupancyDataFrame_1mins = occupancyDataFrame.resample('1min').pad()
#occupancyDataFrame_1mins.head()


# Create a spreadsheet
filename = "occupancy_" + ndays + "days_" + datetime.datetime.today().strftime("%Y-%m-%d_%H-%M") + ".csv"
print("Occupancy Status saved as: " + filename)
occupancyDataFrame.to_csv(filename)  # uncomment this line out and change file name


# Comfort band function. Not needed for actual usage
prob_uncomfortable = np.linspace(1,0,num=100)
z_vals = norm.ppf(prob_uncomfortable/2 +np.ones(len(prob_uncomfortable))/2)
delta_T = z_vals*3.937

plt.plot(prob_uncomfortable*100, delta_T)
xlabel('Percent of Occupants Uncomfortable [%]')
ylabel('Change in Temperature Band [deg C]')
grid()
