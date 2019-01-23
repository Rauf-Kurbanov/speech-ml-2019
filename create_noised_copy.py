import os
import random
import shutil
import numpy as np
import librosa
from argparse import ArgumentParser

def load_audio(file_path, sr=16000):
    data, _ = librosa.core.load(file_path, sr)
    return data, _

noise_dir = "../../bg_noise/"
beeps_dir = noise_dir + "/FRESOUND_BEEPS_gsm/train/"
music_dir = noise_dir + "/AUDIONAUTIX_MUSIC_gsm/train/"


def cycle(data, n):
    return np.concatenate([data for i in range(n)])
def cycle_for_length_with_silence(data, length, silence_ratio):
    silence = np.zeros(int(data.shape[0] * silence_ratio))
    data_with_silence = np.concatenate([data, silence])
    k = length // data_with_silence.shape[0] + 1
    cycled_data = cycle(data_with_silence, k)
    return cycled_data[:length]

def modify_sound(data):
    beep = random.choice(beeps)
    music = random.choice(musics)
    cycled_beep = cycle_for_length_with_silence(beep, data.shape[0], 1.0)
    cycled_music = cycle_for_length_with_silence(music, data.shape[0], 0.1)
    return 0.2 * cycled_beep + 0.2 * cycled_music + 0.6 * data

def process_dir(dirname):
    dst_dir = dirname.strip('/') + "_noised/"
    shutil.copytree(dirname, dst_dir)
    files = os.listdir(dst_dir)
    for filename in files:
        if filename[-4:] != '.wav' and filename[-5:] != '.flac':
            os.remove(dst_dir + filename)
    files = os.listdir(dst_dir)
    for filename in files:
        sr = 16000
        data = load_audio(dst_dir + '/' + filename, sr)[0]
        noised_data = modify_sound(data)
        librosa.output.write_wav(y=noised_data, path=dst_dir + '/' + filename, sr=sr)
        
        


def main():
    beeps_files = os.listdir(beeps_dir)
    music_files = os.listdir(music_dir)
    beeps = [load_audio(beeps_dir + filename)[0] for filename in beeps_files]
    musics = [load_audio(music_dir + filename)[0] for filename in music_files[10:15]]

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help="path to input dir with wav/flac audio files\n" +\
                                               "\toutput is <input dir>_noised")
    args = parser.parse_args()
    process_dir(args.input)
