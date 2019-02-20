import os
import random
import shutil
import numpy as np
import librosa
from argparse import ArgumentParser

def load_audio(file_path, sr=16000):
    data, _ = librosa.core.load(file_path, sr)
    return data, _


def cycle(data, n):
    return np.concatenate([data for i in range(n)])
def cycle_for_length_with_silence(data, length, silence_ratio):
    silence = np.zeros(int(data.shape[0] * silence_ratio))
    data_with_silence = np.concatenate([data, silence])
    k = length // data_with_silence.shape[0] + 1
    cycled_data = cycle(data_with_silence, k)
    return cycled_data[:length]

def modify_sound(data, music_coef, beep_coef, beeps, musics):
    if music_coef + beep_coef > 1:
        raise Exception("beep_coef + music_coef have to be less than 1")
    beep = random.choice(beeps)
    music = random.choice(musics)
    cycled_beep = cycle_for_length_with_silence(beep, data.shape[0], 1.0)
    cycled_music = cycle_for_length_with_silence(music, data.shape[0], 0.1)
    return beep_coef * cycled_beep + music_coef * cycled_music + (1 - beep_coef - music_coef) * data

def process_dir(dirname, music_coef, beep_coef, beeps, musics):
    dst_dir = dirname.strip('/') + "_noised/"
    shutil.copytree(dirname, dst_dir)
    files = os.listdir(dst_dir)
    for filename in files:
        if filename[-4:] != '.wav' and filename[-5:] != '.flac':
            continue
        sr = 16000
        data = load_audio(os.path.join(dst_dir, filename), sr)[0]
        noised_data = modify_sound(data, music_coef, beep_coef, beeps, musics)
        librosa.output.write_wav(y=noised_data, path=os.path.join(dst_dir, filename), sr=sr)
        
        


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help="path to input dir with wav/flac audio files\n" +\
                                               "\toutput is <input dir>_noised",
                        required=True)
    parser.add_argument('-d',
                        '--noise_dir',
                        help="path to dir with bg_noise datasets",
                        default="../bg_noise/")
    parser.add_argument('-n',
                        '--noise',
                        help="path to dir with noise dataset (relative, inside the noise_dir)",
                        default="FRESOUND_BEEPS_gsm/train/")
    parser.add_argument('-m',
                        '--music',
                        help="path to dir with music dataset (relative, inside the noise_dir)",
                        default="AUDIONAUTIX_MUSIC_gsm/train/")
    parser.add_argument('--music_coef',
                        type=float,
                        default=0.2)
    parser.add_argument("--beep_coef",
                        type=float,
                        default=0.2)
    args = parser.parse_args()

    beeps_dir = os.path.join(args.noise_dir, args.noise)
    music_dir = os.path.join(args.noise_dir, args.music)

    beeps_files = os.listdir(beeps_dir)
    music_files = os.listdir(music_dir)
    beeps = [load_audio(os.path.join(beeps_dir, filename))[0] for filename in beeps_files]
    musics = [load_audio(os.path.join(music_dir + filename))[0] for filename in music_files[10:15]]
    process_dir(args.input, args.music_coef, args.beep_coef, beeps, musics)

if __name__ == '__main__':
    main()