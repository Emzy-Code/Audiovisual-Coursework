import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def magAndPhase(speechFrame):
    hamming_window = np.hamming(len(speechFrame))
    xF = np.fft.fft(speechFrame * hamming_window)
    mag = np.abs(xF)
    mag = mag[0:int(len(speechFrame) / 2)]
    phase = np.angle(xF)
    return mag, phase


def linearRectangularFilterbank(magspec, fbankNum):
    fbank = np.zeros(fbankNum)
    for i in range(len(fbank)):
        fbank[i] = sum(magspec[i * 32:i + 32])
    return fbank

def file_feature_extraction(soundfile):
    x, samplerate = sf.read(soundfile)
    length = len(x)
    print(length)
    numFrames = 250
    frameLength = int(len(x) /numFrames)

    print(frameLength)
    all_magnitude_spectra = []
    all_phase_spectra = []
    for frame in range(0, numFrames):
        firstSample = frame * frameLength  # floor(frameLength/2) <- replace with frameLength to make frames overlap
        lastSample = firstSample + frameLength
        shortTimeFrame = x[firstSample:lastSample]
        magSpec, phaseSpec = magAndPhase(shortTimeFrame)
        all_magnitude_spectra.append(magSpec)
        all_phase_spectra.append(phaseSpec)
    return all_magnitude_spectra, all_phase_spectra




mag, phase = file_feature_extraction('./data/a.wav')
