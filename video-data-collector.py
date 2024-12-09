## video-data-collector.py uses obs websocket api to record video data


import time
import math

import obswebsocket
from obswebsocket import requests, obsws
from dotenv import load_dotenv
import os

# Connect to OBS WebSocket
load_dotenv()
password = os.getenv('PASSWORD')
ws = obsws("localhost", 4444)
ws.connect()
load_dotenv()
record_count = 0

classes = sorted(['Muneeb',
                  'Zachary',
                  'Sebastian',
                  'Danny',
                  'Louis',
                  'Ben',
                  'Seb',
                  'Ryan',
                  'Krish',
                  'Christopher',
                  'Kaleb',
                  'Konark',
                  'Amelia',
                  'Emilija',
                  'Naima',
                  'Leo',
                  'Noah',
                  'Josh',
                  'Joey',
                  'Kacper'])
print(sorted(classes))
nameCount = 0

cycleValidation = False
while not cycleValidation:
    cycles = input("How many cycles: ")
    if cycles.isdigit():
        cycleValidation = True
    else:
        print("Please enter an integer. ")

for i in range(int(cycles)):
    for j in range(20):
        record_count += 1


        print("nameCount: ",nameCount)
        print(f"Video {record_count} \n Name: {classes[nameCount]}")
        print(f"Cycle: {i+1}")
        string = f"{classes[nameCount]}_{i}"
        print("Name: ",string)
        if nameCount < 19:
            nameCount += 1
        else:
            nameCount = 0
        #ws.call(requests.SetFilenameFormatting(**{f'filename-formatting': string}))
        #response = ws.call(requests.StartRecording())
        #time.sleep(5)  # Record for 5 seconds
        # Stop recording
        #ws.call(requests.StopRecording())
        #time.sleep(1)  # Short delay before starting again
