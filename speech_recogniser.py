import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras import Sequential
from keras._tf_keras.keras.layers import Dense, Activation, Flatten, Conv2D, InputLayer, MaxPooling2D
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

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
LE = LE.fit(classes)
labels = to_categorical(LE.transform(labels))
data = np.array(data)
data = data / np.max(data)

X,y = data,labels
model = load_model("my_model.keras")
predicted_probs = model.predict(X, verbose=0)
predicted = np.argmax(predicted_probs, axis=1)
actual = np.argmax(y, axis=1)
print(np.unique(predicted))
accuracy = metrics.accuracy_score(actual, predicted)
print(f'Accuracy: {accuracy * 100}%')
print(X[0, :, :])
predicted_prob = model.predict(np.expand_dims(X[0, :, :],
                                              axis=0), verbose=0)
predicted_id = np.argmax(predicted_prob, axis=1)
predicted_class = LE.inverse_transform(predicted_id)
predicted_classes = []


print(predicted)