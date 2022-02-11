import numpy as np
import soundfile as sf
import librosa.display
import librosa
import pydub
import matplotlib.pyplot as plt
import os
import random


# Adding random noise
# noise factor is used to monitor how much of it we want
def add_white_noise(signal, noise_factor):
    noise = np.random.normal(0,signal.std(), 
                             signal.size)  # Noise vector created using Gaussian Distribution
    
    augmented_signal = signal + noise * noise_factor

    return augmented_signal


# Time stretch
def time_stretch(signal , stretch_rate):

    return librosa.effects.time_stretch(signal, stretch_rate)


# Scaling Pitch
# num_semitones is used to increase or decrease it
def pitch_scale(signal , sr, num_semitones):

    return librosa.effects.pitch_shift(signal ,sr , num_semitones)


# Polarity Inversion
def invert_polarity(signal):

    return signal*(-1)



# Random Gain
def random_gain(signal, min_gain_factor,
                max_gain_factor):
    
    gain_factor = random.uniform(min_gain_factor,
                                  max_gain_factor)
    
    return signal * gain_factor
