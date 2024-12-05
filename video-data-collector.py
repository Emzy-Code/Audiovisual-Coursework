## video-data-collector.py uses obs websocket api to record video data


import time

import obswebsocket
from obswebsocket import requests, obsws
from dotenv import load_dotenv
import os

# Connect to OBS WebSocket
load_dotenv()
password = os.getenv('PASSWORD')
print(password)
ws = obsws("localhost", 4455)
ws.connect()
load_dotenv()
record_count = 0

try:
    while True:
        record_count += 1
        print(f"Video {record_count}")
        ws.call(requests.SetFilenameFormatting("TESTNAME" + str(record_count)))
        ws.call(requests.StartRecording())
        time.sleep(5)  # Record for 5 seconds
        # Stop recording
        ws.call(requests.StopRecording())
        time.sleep(1)  # Short delay before starting again
except KeyboardInterrupt:
    pass
finally:
    ws.disconnect()
