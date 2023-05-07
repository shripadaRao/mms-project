""" generate various x seconds clip of both sawing and non-sawing sound.
    introduce random(or normal distribution) level of noise to audio clips.
    vary audio intensity, ideally follow normal distribution for this."""

"""varying the audio intensity is very straight forward, just multiply a number between 0-1"""

#IMPORTS
from scipy.io import wavfile
from scipy.io.wavfile import write
import numpy as np
import random
import time
import os


#utils
def readWavFile(file):
    sampleRate, data = wavfile.read(file)
    return sampleRate,data

def writeWavFile(fileName,sampleRate,data):
    return write(fileName,sampleRate, data.astype(np.int16))

def getNormalDistNum():
    # prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    # return prob_density
    return random.gauss(0.4, 0.15)


def applyNoiseTorAudio(audio_data):
    return audio_data + np.random.normal(0,0.1,audio_data.shape)

def tranformAudioData(audio_data):
    intensity_varied_audio_data = data*getNormalDistNum()
    final_audio_data = applyNoiseTorAudio(intensity_varied_audio_data)
    return final_audio_data


# sampleRate, data = readWavFile('sawingSample.wav')
# data = data*0.1
# writeWavFile('loweredAudioIntensity.wav',sampleRate,data)

if __name__ == "__main__":
    count = 11
    for file in os.listdir('dataset/ambient/'):
        print(file)
        sampleRate, data = readWavFile('dataset/ambient/'+file)

        audio_samples = 1000 - 1
        for i in range(audio_samples):
            transformed_audio_data = tranformAudioData(data)
            writeWavFile('dataset/ambient/file'+str(count)+'.wav', sampleRate, transformed_audio_data)
            count += 1
