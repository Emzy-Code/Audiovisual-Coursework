# -Created 23 Oct, Emilija
# This file is used to record audio and automatically store it within the training_data folder for later use



import sounddevice as sd
import soundfile as sf

fs = 16000
seconds = 5
r = sd.rec(seconds * fs, samplerate=fs, channels=1)
sd.wait()
sf.write('training_data/audio/Kacper19.wav', r, fs) #change wav file name using naming convension above
#r2, fs2 = sf.read('muneeb/Muneeb_00.wav', dtype='float32')
#sd.play(r2,fs2)
#sd.wait()
#print("yes")