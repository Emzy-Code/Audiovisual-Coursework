import glob
import matplotlib.pyplot as plt
from acoustic_modelling import model
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.utils import to_categorical

max_frames = 21
labels = []
data = []
for mfcc_file in sorted(glob.glob('test_data/mfccs/*.npy')):
    mfcc_data = np.load(mfcc_file)
    mfcc_data = np.pad(mfcc_data, ((0, 0), (0, max_frames - mfcc_data.shape[1])))
    data.append(mfcc_data)
    stemFilename = (Path(os.path.basename(mfcc_file))).stem
    label = stemFilename.split('_')
    labels.append(label[0])
LE = LabelEncoder()
classes = ['Muneeb',
'Zachary',
'Sebastian',
'Danny',
'Louis',
'Ben',
'Seb',
'Ryan',
'Krish',
'Christopher',
'Kaleb',
'Konark',
'Amelia',
'Emilija',
'Naima',
'Leo',
'Noah',
'Josh',
'Joey',
'Kacper']
LE = LE.fit(classes)
labels = to_categorical(LE.transform(labels))
X,y = data,labels
predicted_X, predicted_y = model.predict(X,y)
