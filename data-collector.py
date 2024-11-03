# -Created 23 Oct, Emilija
# This file is used to record audio and automatically place it within audio folder
# each file should have the format of name_number.wav (example: ryan_0.wav)
# sound device used - :
# voice actor - Emilija:


import sounddevice as sd
import soundfile as sf

fs = 16000
seconds = 5
r = sd.rec(seconds * fs, samplerate=fs, channels=1)
sd.wait()
sf.write('kaleb/Christopher_09.wav', r, fs) #change wav file name using naming convension above
#r2, fs2 = sf.read('muneeb/Muneeb_00.wav', dtype='float32')
#sd.play(r2,fs2)
#sd.wait()
#print("yes")