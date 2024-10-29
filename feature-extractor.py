import numpy as np
import matplotlib.pyplot as plt

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
