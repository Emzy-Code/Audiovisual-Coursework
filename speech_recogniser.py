# >>>>>>Speech Recogniser <<<<<<<
#
#  Uses model created in acoustic_modelling.py to predict names spoken
#  Labelling method: Unlike in "training_data" , the last number indicates data order
#  Input: ./test_data/mfccs
#  Required file: ./my_model.keras
#  ((Tip: run "acousting_modelling.py" if non-existent or model changes are made))
#  Output: Predicted names, accuracy score
#

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from numba.core.cgutils import sizeof
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.utils import to_categorical
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
classes = sorted(classes)
print("classes: ", classes)
max_frames = 21
labels = []
data = []
orderedList = []
unorderedList = []

for mfcc_file in sorted(glob.glob('test_data/mfccs/*.npy')):
    mfcc_data = np.load(mfcc_file)
    mfcc_data = np.pad(mfcc_data, ((0, 0), (0, max_frames - mfcc_data.shape[1])))
    data.append(mfcc_data)
    stemFilename = (Path(os.path.basename(mfcc_file))).stem
    orderedList.append(stemFilename)
    unorderedList.append(stemFilename)
    label = stemFilename.split('_')
    labels.append(label[0])

if len(labels)>0:
    positions = []
    finalPositions = []

    def get_number(order):
        num = order.split('_')[1]
        print("num: ", num)
        return int(num)

    print("order: ", orderedList)
    orderedList = (sorted(orderedList, key = lambda order: get_number(order)))
    print("sorted: ", orderedList)
    for i in range(len(orderedList)):
        positions.append(i)

    print("labels: ", labels)
    print("Final Pos: " ,finalPositions)
    LE = LabelEncoder()
    LE = LE.fit(classes)
    labels = to_categorical(LE.transform(labels))
    data = np.array(data)
    data = data / np.max(data)

    X,y = data,labels
    model = load_model("my_model.keras")
    predicted_probs = model.predict(X, verbose=0)
    predicted = np.argmax(predicted_probs, axis=1)
    print("Predicted: ", predicted)
    actual = np.argmax(y, axis=1)
    #print(np.unique(predicted))
    accuracy = metrics.accuracy_score(actual, predicted)
    print(f'Accuracy: {accuracy * 100}%')
    #print(X[0, :, :])
    predicted_prob = model.predict(np.expand_dims(X[0, :, :],
                                                  axis=0), verbose=0)
    predicted_id = np.argmax(predicted_prob, axis=1)


    predicted_classes = []

    for i in range(len(predicted)):
        predicted_classes.append(classes[predicted[i]])

    result=[]
    for i in range(len(unorderedList)):
        result.append(0)
        finalPositions.append(orderedList.index(unorderedList[i]))

    print(positions)
    print(finalPositions)


    for i in range(len(predicted_classes)):
        result[finalPositions[i]] = predicted_classes[i]

    print(result)
else:
    print("No audio files. Please re-run.")
