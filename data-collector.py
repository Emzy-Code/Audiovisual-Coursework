# -Created 23 Oct, Emilija
# This file is used to record data and automatically place it within data folder
# each file should have the format of name(0-9)(0-9).wav (example: ryan00.wav)
# sound device used - :
# voice actor - Emilija:

import sounddevice as sd
import soundfile as sf

fs = 16000
seconds = 5
r = sd.rec(seconds * fs, samplerate=fs, channels=1)
sd.wait()
sf.write('./data/muneeb01.wav', r, fs) #change wav file name using naming convension above
