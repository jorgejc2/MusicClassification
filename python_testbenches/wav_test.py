import sys
sys.path.append('../build/pybind_modules')
import numpy as np
import wav_module as wav

print("Imports succesful")

wav_path = "../music_samples_wav/chicago.wav"

wav_samples = wav.wavsamples(wav_path)

print("wav_samples with size {} and type {}".format(len(wav_samples), type(wav_samples[0])))

print(wav_samples[:10])