# -Created 23 Oct, Emilija
# This file is used to record audio and automatically store it within the training_data folder for later use

import sounddevice as sd
import soundfile as sf

fs = 48000
seconds = 5
r = sd.rec(seconds * fs, samplerate=fs, channels=1)
sd.wait()
sf.write('test_data/audio/Muneeb_7.wav', r, fs) #change wav file name using naming convention above
#r2, fs2 = sf.read('training_data/audio/Muneeb_06.wav', dtype='float32')
#sd.play(r2,fs2)
#sd.wait()