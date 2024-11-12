# used to resample data

import glob
import librosa
import soundfile as sf

for audiofile in sorted(glob.glob('training_data/audio/*.wav')):
    y, fs = sf.read(audiofile, dtype='float32')
    x = librosa.resample(y, orig_sr=fs, target_sr=48000)
    sf.write(audiofile, x, 48000)
