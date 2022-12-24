import os 
from pydub import AudioSegment
from tqdm import tqdm 

"""
This program is meant to convert mp3 files to wav files
"""

# obtain all mp3 files from the following path
cwd = os.getcwd()
mp3_path = os.getcwd() + '/music_samples_mp3'

mp3_files = os.listdir(mp3_path)

# convert each file and place them into music_samples_wav
for file in tqdm(mp3_files):
    src = 'music_samples_mp3/' + file
    des = 'music_samples_wav/' + file.replace('.mp3','.wav')
    sound = AudioSegment.from_mp3(src)
    sound.set_channels(1)
    sound = sound.set_frame_rate(16000)                
    sound = sound.set_channels(1)    
    sound.export(des, format="wav")