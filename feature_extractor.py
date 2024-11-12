# >>>>>>>>>>>>>>>> feature_extractor - converts audio files to mfccs
# Input: /audio/*.wavs  (/training_data or /test_data)
# Output: /mfccs/*.npys (/training_data or /test_data, and same format)
# For commented out example below:  import matplotlib.pyplot as plt


import numpy as np
import soundfile as sf
from math import floor
import glob
from scipy.fftpack import dct


def byFrameSpectraCalculator(x, frameLength):  # calculates magnitudes by frame
    numSamples = len(x)
    numFrames = floor(numSamples / frameLength)
    all_magnitudes = []
    for frame in range(0, numFrames):
        firstSample = frame * frameLength
        # floor(frameLength/2) <- replace with frameLength to make frames overlap
        lastSample = firstSample + frameLength
        shortTimeFrame = x[firstSample:lastSample]
        magSpec, phaseSpec = magAndPhase(shortTimeFrame)
        all_magnitudes.append(magSpec)
    return all_magnitudes


def mfccVectors(soundfile):  # creates mfcc vectors for a sound file
    x, samplerate = sf.read(soundfile)

    all_magnitudes = byFrameSpectraCalculator(x, 960)
    mel_vector = mel_vector_creator(20, 480, 48000)

    mel_magnitudes = np.dot(all_magnitudes, mel_vector.T)
    mel_magnitudes = np.log(mel_magnitudes + 0.000000000000000001)

    mel_magnitudes = dct(mel_magnitudes, type=2, norm='ortho')
    return mel_magnitudes


def mfccFileCreator():  ## creates mfcc files for entire folder
    for audiofile in sorted(glob.glob('test_data/audio/*.wav')):
        mfccData = mfccVectors(audiofile)
        mfccName = audiofile.removeprefix('test_data/audio').replace(".wav", ".npy")
        filepath = 'test_data/mfccs/' + mfccName
        print(filepath)
        np.save(filepath, mfccData)


def mel_vector_creator(triangles, mag_len, sample_rate):  # creates mel vector
    frequency_resolution = sample_rate / mag_len
    frequencies = np.arange(mag_len * frequency_resolution)
    min_frequency = hz_to_mel(frequencies[0])
    max_frequency = hz_to_mel(frequencies[-1])
    # print(max_frequency)
    points = np.linspace(min_frequency, max_frequency, triangles + 2)
    for i in range(len(points)):
        points[i] = mel_to_hz(points[i])
    # print(spacing)
    points = np.floor(np.array(points) / frequency_resolution)
    # print("Points" , points)

    basisVector = np.zeros((triangles, mag_len))
    for k in range(mag_len):
        for m in range(1, len(points) - 1):
            if k < points[m - 1] or k > points[m + 1]:
                # array.append(0)
                basisVector[m - 1, k] = 0
            if points[m - 1] <= k <= points[m]:
                # array.append((k - points[m-1]) / (points[m] - points[m-1]))
                basisVector[m - 1, k] = (k - points[m - 1]) / (points[m] - points[m - 1])
            if k >= points[m] and k <= points[m + 1]:
                # array.append((points[m+1] - k) / (points[m+ 1] - points[m]))
                basisVector[m - 1, k] = (points[m + 1] - k) / (points[m + 1] - points[m])
    return basisVector


def magAndPhase(speechFrame):
    hamming_window = np.hamming(len(speechFrame))
    xF = np.fft.fft(speechFrame * hamming_window)
    mag = np.abs(xF)
    mag = mag[0:int(len(mag) / 2)]
    phase = np.angle(xF)
    return mag, phase


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def linearRectangularFilterbank(magspec, fbankNum):  # unused linear filterbank
    fbank = np.zeros(fbankNum)
    for i in range(len(fbank)):
        fbank[i] = sum(magspec[i * 32:i + 32])
    return fbank


mfccFileCreator()

### EXAMPLE MFCC GRAPH

# mfcc_path = 'training_data/mfccs/Amelia_06.npy'
# mfcc  = np.load(mfcc_path)
# plt.imshow(mfcc)
# plt.show()
