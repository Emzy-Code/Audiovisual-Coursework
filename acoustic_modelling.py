import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path
import os
from sklearn import metrics
from keras._tf_keras.keras.utils import to_categorical
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Activation, Flatten, Conv2D, InputLayer, MaxPooling2D
from tf_keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def create_model():
    numClasses = 6
    model = Sequential()
    model.add(InputLayer(input_shape=(156, 21, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(numClasses))
    model.add(Activation('softmax'))
    return model


data = []
labels = []
max_frames = 21
for mfcc_file in sorted(glob.glob('training_data/mfccs/*.npy')):
    mfcc_data = np.load(mfcc_file)
    mfcc_data = np.pad(mfcc_data, ((0, 0), (0, max_frames - mfcc_data.shape[1])))
    # if mfcc_data.shape[1] > max_frames:
    #    max_frames = mfcc_data.shape[1]
    data.append(mfcc_data)

    stemFilename = (Path(os.path.basename(mfcc_file))).stem
    label = stemFilename.split('_')
    labels.append(label[0])
labels = np.array(labels)
print(labels)
data = np.array(data)
data = data / np.max(data)

LE = LabelEncoder()
classes = [
'Muneeb',
#'Zachary',
'Sebastian',
'Danny',
'Louis',
'Ben'
    ,
'Seb',
'Ryan',
'Krish',
'Christopher'
    #,
#'Kaleb',
#'Konark',
#'Amelia',
#'Emilija',
#'Naima',
#'Leo',
#'Noah',
#'Josh',
#'Joey',
#'Kacper',
]
LE = LE.fit(classes)
labels = to_categorical(LE.transform(labels))
X_train, X_tmp, y_train, y_tmp = train_test_split(data, labels, test_size=0.2, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)
learning_rate = 0.01
model = create_model()
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer=Adam(learning_rate=learning_rate))
model.summary()

num_epochs = 30
num_batch_size = 20

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=num_batch_size, epochs=num_epochs,
                    verbose=1)
model.save_weights('name_classification.h5')

model = create_model()

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer=Adam(learning_rate=learning_rate))
model.load_weights('name_classification.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

predicted_probs = model.predict(X_test, verbose=0)
predicted = np.argmax(predicted_probs, axis=1)
actual = np.argmax(y_test, axis=1)
accuracy = metrics.accuracy_score(actual, predicted)
print(f'Accuracy: {accuracy * 100}%')
print(X_test[0, :, :])
predicted_prob = model.predict(np.expand_dims(X_test[0, :, :],
                                              axis=0), verbose=0)
predicted_id = np.argmax(predicted_prob, axis=1)
predicted_class = LE.inverse_transform(predicted_id)
print(predicted)
confusion_matrix = metrics.confusion_matrix(
    np.argmax(y_test, axis=1), predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
cm_display.plot()
plt.show()
model = create_model()

