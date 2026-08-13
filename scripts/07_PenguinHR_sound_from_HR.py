"""
LINKING HEART RATE DATA TO SOUND
Created on Tue May 18 10:35:14 2021

@author: Jessica Kendall-Bar
"""
#%%
# import libraries
import os
import pandas as pd
from psychopy import prefs, core, sound

# UPDATE ME
data_path = "G:/My Drive/Visualization/Data"

os.chdir(data_path)
os.getcwd()

# From this website https://mbraintrain.com/how-to-set-up-precise-sound-stimulation-with-psychopy-and-pylsl/
# Change the pref libraty to PTB (psychtoolbox) 
prefs.hardware['audioLib'] = 'PTB'
# Set the latency mode to high precision (3)
prefs.hardware['audioLatencyMode'] = 3

sounddur = 0.200 # 200 ms 

# Load in heartbeat sound
badum = sound.Sound('07_HeartBeat_200ms.wav') #sound of heart beating
# swish = sound.Sound('01 Tail Noise.wav') #sound of tail swishing back and forth

# Load in heartrate data (with array of interbeat intervals in seconds)
HR_data = pd.read_csv('07_Penguin-Phys_PP_02_HR_interval.csv', sep=",", header=0, squeeze=True)


# After heartbeat plays, wait interval - duration of heartbeat until next.
HR_data['Wait'] = abs(HR_data["Interval"] - sounddur)

# Initializing wait variable with wait durations
wait = HR_data['Wait']

for i in range(wait.first_valid_index(),wait.last_valid_index()): # for all values in wait series
    playback_time = core.getTime() # get current time
    curr_time = core.getTime() - playback_time # get elapsed time
    while curr_time < wait[i]: # until it's time to play next heart beat
        curr_time = core.getTime() - playback_time # continue getting elapsed time
        print("Seconds since last heartbeat: %3.5f" %curr_time)
        core.wait(0.005) # wait 50 milliseconds
    badum.setVolume(1.0)
    badum.play() # play next heartbeat
    core.wait(sounddur) # determined by duration of heartbeat