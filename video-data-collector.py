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
print(password)
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
nameCount = 0
try:
    while True:
        record_count += 1
        nameCount = nameCount + (1 / 20)
        print(f"Video {record_count} \n Name: {classes[math.floor(nameCount)]}")
        string = f"TESTNAME {record_count}"

        response = ws.call(requests.StartRecording())
        time.sleep(5)  # Record for 5 seconds
        # Stop recording
        ws.call(requests.StopRecording())
        ws.call(requests.SetFilenameFormatting(**{f'filename-formatting': string}))
        time.sleep(1)  # Short delay before starting again
except KeyboardInterrupt:
    pass
finally:
    ws.disconnect()
