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
from scipy.io import wavfile 
import librosa
import pydub


path = os.getcwd()

def features_extractor(file_name):
	audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
	mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40) 
	mfccs_scaled_features = np.mean(mfccs_features.T,axis=0) 

	return mfccs_scaled_features

def absolute_file_paths(directory):
	path = os.path.abspath(directory)

	return [entry.path for entry in os.scandir(path) if entry.is_file()]


scream_paths = absolute_file_paths(path + r'\data\positive')
non_scream_paths = absolute_file_paths(path +r'\data\negative')
atmospheric_paths = absolute_file_paths(path +r'\data\atmospheric_data')


def extract_features(scream_paths, non_scream_paths, atmospheric_paths):

    extracted_features = [] #setting labels

    for scream in scream_paths:
        data = features_extractor(scream)
        label = 'scream'
        extracted_features.append([data,label])

    for non_scream in non_scream_paths:
        data = features_extractor(non_scream)
        label = 'non_scream'
        extracted_features.append([data,label])

    for non_scream in atmospheric_paths:
        data = features_extractor(non_scream)
        label = 'atmospheric'
        extracted_features.append([data,label])

    extracted_features_array = np.asarray(extracted_features)

    extracted_features_audio = [i[0] for i in extracted_features_array]
    extracted_features_audio = np.asarray(extracted_features_audio) 

    extracted_features_labels = [i[1] for i in extracted_features_array]
    extracted_features_labels = np.asarray(extracted_features_labels)

    return extracted_features_audio ,extracted_features_labels

extracted_features_audio, extracted_features_labels = extract_features(scream_paths, non_scream_paths, atmospheric_paths)



lb = LabelBinarizer()
labels = lb.fit_transform(extracted_features_labels)
X_train,X_test,y_train,y_test=train_test_split(extracted_features_audio,labels,test_size=0.3,random_state=0)

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





    checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', 
                                   verbose=1, save_best_only=True)
    start = datetime.now()

    history = model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)


    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    predictions = model.predict(x=X_test, batch_size=32)
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


