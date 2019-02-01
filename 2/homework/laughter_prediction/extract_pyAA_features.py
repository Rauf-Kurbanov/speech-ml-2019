from __future__ import print_function

import argparse
import scipy.io.wavfile as wav
import numpy as np
import pandas as pd
from pyAudioAnalysis.audioFeatureExtraction import stFeatureExtraction, mtFeatureExtraction
from sklearn.preprocessing import StandardScaler


def get_features_from_wav(wav_path, sec):
    """
    Samples audio by given time window

    :param wav_path: path to .wav file
    :param sec: float, sampling frame size in sec
    :return: pandas.DataFrame with sampled audio of shape (n_samples, frames_per_sample)
    """
    rate, audio = wav.read(wav_path)

    short_frame = rate * sec
    mt_features = mtFeatureExtraction(audio, rate, mtWin=short_frame * 10, mtStep=short_frame,
                                      stWin=short_frame, stStep=short_frame)
    big_mat = np.vstack([mt_features[0], mt_features[1]]).T
    big_mat = StandardScaler().fit_transform(big_mat)
    big_df = pd.DataFrame(big_mat)
    colnames = ["pyAA{}".format(i) for i in range(big_mat.shape[1])]
    big_df.columns = colnames

    return big_df


def main():
    parser = argparse.ArgumentParser(description='Feature extraction script based on PythonAudioAnalysis features')
    parser.add_argument('--frame_ms', type=int, default=10,
                        help='Length of each frame in ms')
    parser.add_argument('--wav_path', type=str, help='Path to .wav dile')
    parser.add_argument('--feature_save_path', type=str, help='Path to save features .csv file')
    args = parser.parse_args()

    feature_df = get_features_from_wav(args.wav_path, 0.001 * args.frame_ms)
    print("Created features dataframe with shape:", feature_df.shape)

    print("Saving features:", args.feature_save_path)
    feature_df.to_csv(args.feature_save_path, index=False)
    print("Done")

if __name__ == '__main__':
    main()
