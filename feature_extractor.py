import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
from spyder_kernels.utils.lazymodules import scipy
import glob
from scipy.fftpack import dct


def byFrameSpectraCalculator(x, sampleRate, numFrames):
    frameLength = int(len(x) / numFrames)
    all_magnitudes = []

    for frame in range(0, numFrames):
        firstSample = frame * frameLength
        # floor(frameLength/2) <- replace with frameLength to make frames overlap
        lastSample = firstSample + frameLength
        shortTimeFrame = x[firstSample:lastSample]
        magSpec, phaseSpec = magAndPhase(shortTimeFrame)
        all_magnitudes.append(magSpec)
    return all_magnitudes


def mfccVectors(soundfile):
    x, samplerate = sf.read(soundfile)


    numFrames = 156

    frameLength = int(len(x) / numFrames)
    all_magnitudes = byFrameSpectraCalculator(x, samplerate, numFrames)

    print(len(all_magnitudes))
    mel_vector = mel_vector_creator(20, 256, 8000)

    mel_magnitudes = np.dot(all_magnitudes, mel_vector.T)
    mel_magnitudes = np.log(mel_magnitudes + 0.000000000000000001)


    mel_magnitudes = dct(mel_magnitudes,type=2,norm='ortho')
    return mel_magnitudes


def mfccFileCreator():
    for audiofile in sorted(glob.glob('training_data/audio/*.wav')):
        mfccData = mfccVectors(audiofile)
        print(audiofile)
        mfccName = audiofile.removeprefix('training_data/audio').replace(".wav", ".npy")
        filepath = 'training_data/mfccs/' + mfccName
        np.save(filepath, mfccData)


def mel_vector_creator(triangles, mag_len, sample_rate):
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
            if k >= points[m - 1] and k <= points[m]:
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


def linearRectangularFilterbank(magspec, fbankNum):
    fbank = np.zeros(fbankNum)
    for i in range(len(fbank)):
        fbank[i] = sum(magspec[i * 32:i + 32])
    return fbank


mfccFileCreator()

