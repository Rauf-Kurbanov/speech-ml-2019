from enum import Enum
import numpy as np
import librosa
from os import listdir
import sys

def load_audio_sec(file_path, sr=16000):
    data, _ = librosa.core.load(file_path, sr)
    if len(data) > sr:
        data = data[:sr]
    else:
        data = np.pad(data, pad_width=(0, max(0, sr - len(data))), mode="constant")
    return data


class SupportedFormats(Enum):
    WAV = ".wav"
    FLAC = ".flac"


def add_noise(source_file_full_path: str, to_save_dir: str, source_file_name: str,
              noise_beep: str, noise_background: str,
              coef: float, format: SupportedFormats):
    data = load_audio_sec(source_file_full_path)
    noise_beep = load_audio_sec(noise_beep)
    noise_backgroud = load_audio_sec(noise_background)

    data_with_noise = data + coef * noise_beep
    data_with_noise = data_with_noise + coef * noise_backgroud

    if format == SupportedFormats.WAV:
        librosa.output.write_wav(to_save_dir + source_file_name[:-4] + "_with_noise.wav",
                                 data_with_noise, sr=16000)
    elif format == SupportedFormats.FLAC:
        librosa.output.write_wav(to_save_dir + source_file_name[:-5] + "_with_noise.flac",
                                 data_with_noise, sr=16000)


if __name__ == '__main__':

    source_dir = sys.argv[1]
    source_files = listdir(source_dir)
    for file in source_files:
        if file.endswith(".wav"):
            add_noise(source_dir + file, "output/", file,
                      "noise/15482_45914-lq.wav",
                      "noise/noise-free-sound-0030.wav",
                      0.5, SupportedFormats.WAV)
        elif file.endswith(".flac"):
            add_noise(source_dir + file, "output/", file,
                      "noise/15482_45914-lq.wav",
                      "noise/noise-free-sound-0030.wav",
                      0.5, SupportedFormats.FLAC)
