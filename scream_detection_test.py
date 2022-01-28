import os
import numpy as np
from scipy.io import wavfile 
import librosa
import pydub
from keras.models import load_model
#from sklearn.metrics import classification_report




def predict(file, model):

	model = load_model(model)
	
	audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 

	mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
	mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
	mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)

	labels = ['atmospheric', 'non_scream', 'scream']
	labels.sort()
	
	predict_x=model.predict(mfccs_scaled_features) 
	classes_x=np.argmax(predict_x,axis=1)

	return labels[classes_x[0]]

path = os.getcwd()
model_path = path +  r'\saved_models\audio_classification.hdf5'
file_path  = path +  r'/data/test_positive/scream.wav'

predicted_list = predict(file_path, model_path)
print(predicted_list)
