import tensorflow as tf
#from tensorflow.keras.layers import Input, Lambda, Conv2D, BatchNormalization
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from datetime import datetime 
from sklearn.preprocessing import LabelBinarizer

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam


#from IPython.display import Audio
from matplotlib import pyplot as plt
import os
import numpy as np
import random
from scipy.io import wavfile 
import librosa
import pydub

path = os.getcwd()

def absolute_file_paths(directory): #setting path
    path = os.path.abspath(directory)
    return [entry.path for entry in os.scandir(path) if entry.is_file()]

scream_paths = absolute_file_paths(path + r'\data\positive')
non_scream_paths = absolute_file_paths(path +r'\data\negative')
atmospheric_paths = absolute_file_paths(path +r'\data\negative')


all_paths = []
all_labels = []

for scream in scream_paths:
    label = 'scream'
    all_paths.append(scream)
    all_labels.append(label)

for non_scream in non_scream_paths:
    label = 'non_scream'
    all_paths.append(non_scream)
    all_labels.append(label)

for atmos in atmospheric_paths:
    label = 'atmospheric'
    all_paths.append(atmos)
    all_labels.append(label)


lb = LabelBinarizer()
labels = lb.fit_transform(all_labels)

### Train Test Split
X_train,X_test,y_train,y_test=train_test_split(all_paths ,labels ,test_size=0.3,random_state=0)

def get_signal(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    return audio, sample_rate

X_train_signal = []
for audio in X_train:
    signal , sr =  get_signal(audio)
    X_train_signal.append([signal, sr])

def data_aug(signal, sample_rate):

    noise_factor = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    stretch_rate = [0.6, 0.65, 0.7, 0.75, 0.8]
    num_semitones = [-2,-1, 0, 1, 2]

    noise = np.random.normal(0,signal.std(), signal.size)  # Noise vector created using Gaussian Distribution

    augmented_1 =  signal + noise * random.choice(noise_factor)

    augmented_2 =  librosa.effects.time_stretch(augmented_1, random.choice(stretch_rate))

    augmented_3 =  librosa.effects.pitch_shift(augmented_2 , sample_rate , random.choice(num_semitones))

    return augmented_3, sample_rate

augmented_signals = []
for signals in X_train_signal:

    signal = signals[0]
    sample_rate = signals[1]
    augmented_signal, sample_rate = data_aug(signal, sample_rate)
    augmented_signals.append([augmented_signal, sample_rate])

    augmented_signals = np.asarray(augmented_signals)

X_train_signal   = np.asarray(X_train_signal)

#print(augmented_signals.shape)
#print(X_train_signal.shape)

new_X_train = np.concatenate((augmented_signals, X_train_signal), axis=0)
#print(new_X_train.shape)

y_train_1 = np.copy(y_train)
new_y_train = np.concatenate((y_train, y_train_1), axis=0)


def features_extractor(audio, sample_rate):
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40) #spectogram? feature engineering for audio
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0) #scaling, mean , standardization?
    
    return mfccs_scaled_features

X_train_features = []
for x in new_X_train:
    #new_X_train_features = features_extractor()
    audio = x[0]
    sample_rate = x[1]
    mfccs_scaled_features = features_extractor(audio, sample_rate)
    X_train_features.append(mfccs_scaled_features)


X_train_features = np.asarray(X_train_features)

def features_extractor_test(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') #loading audio files, #sample rate- numerical int value of each music file
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40) #spectogram? feature engineering for audio
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0) #scaling, mean , standardization?
    
    return mfccs_scaled_features

X_test_features = []
for x in X_test:
    data = features_extractor_test(x)
    X_test_features.append(data)

X_test_features = np.asarray(X_test_features)


num_labels = 3
num_epochs = 500
num_batch_size = 32


def model():

    model=Sequential() 
    ###first layer
    model.add(Dense(100,input_shape=(40,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    ###second layer
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    ###third layer
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    ###final layer
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')





    checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification_data_aug.hdf5', 
                                   verbose=1, save_best_only=True)
    start = datetime.now()

    history = model.fit(X_train_features, new_y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test_features, y_test), callbacks=[checkpointer],
          verbose=1, shuffle = True)


    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    predictions = model.predict(x=X_test_features, batch_size=32)
    report = classification_report(y_test.argmax(axis=1),
        predictions.argmax(axis=1), target_names=lb.classes_)

    return history, report

history, matrix = model()

print(matrix)

N = num_epochs
plt.style.use("ggplot")

plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.ylim(0,1.5)
plt.xlim(0,505)
plt.title("Training Loss and Accuracy on Dataset")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
#plotPath = r'E:\Rohan\Sem 7\Minor Project\Major Project\Model'
plt.show()
