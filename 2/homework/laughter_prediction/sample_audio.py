import numpy as np
import pandas as pd
import scipy.io.wavfile as wav


def sample_audio_by_frames(audio, frame_size):
    """
    Samples audio by chunks of fixed size

    :param audio: array-like, representation of raw audio
    :param frame_size: int, number of frames per chunk
    :return: Pandas DataFrame, each chunk represented by row
    """
    last = len(audio) // frame_size * frame_size
    samples = np.array([np.array(audio[i:i + frame_size])
                        for i in range(0, last, frame_size)])
    return pd.DataFrame(samples)


def sample_wav_by_time(wav_path, frame_sec):
    """
    Samples audio by chunks of fixed time

    :param wav_path: path to wav file to sample
    :param frame_sec: int, length of each chunk in seconds
    :return: Pandas DataFrame, each chunk represented by row
    """
    rate, audio = wav.read(wav_path)
    frame_size = int(rate * frame_sec)
    return sample_audio_by_frames(audio, frame_size)


def main():
    wav_path = "/home/kurbanov/Data/vocalizationcorpus/data/S0458.wav"
    df = sample_wav_by_time(wav_path, frame_sec=0.01)
    print(df.shape)
    print(df.head())


if __name__ == '__main__':
    main()
