import os
import tempfile
import librosa
import numpy as np

import pandas as pd

from homework.laughter_classification.sspnet_data_sampler import SSPNetDataSampler


class FeatureExtractor:
    def extract_features(self, wav_path: str, data_path: str) -> pd.DataFrame:
        """
        Extracts features for classification ny frames for .wav file

        :param wav_path: string, path to .wav file
        :param data_path: string, path to all files and lables.txt
        :return: pandas.DataFrame with features of shape (n_chunks, n_features)
        """
        data_sampler = SSPNetDataSampler(data_path)
        df_wav = data_sampler.df_from_file(wav_path, 1)

        sr = 16_000
        melspec_features = 128
        mfcc_features = 13

        M = np.zeros((11, (melspec_features + mfcc_features + 1)))
        i = 0
        for _, row in df_wav.iterrows():
            S = librosa.feature.melspectrogram(np.array(row[:sr], dtype=np.float32), sr=sr,
                                               n_fft=sr, hop_length=(sr + 1), n_mels=melspec_features)

            log_S = librosa.power_to_db(S, ref=np.max)
            mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=mfcc_features)
            t = np.concatenate((mfcc, log_S)).T
            a = np.array(row[-2:-1]).reshape(1, 1)
            t = np.concatenate((t, a), axis=1)
            M[i] = t[0]
            i += 1

        col_names = ["{}".format(i) for i in range(melspec_features + mfcc_features)]
        col_names.append("IS_LAUGHTER")

        return pd.DataFrame(M, columns=col_names)

    def extract_features474(self, wav_path: str, data_path: str) -> pd.DataFrame:
        """
        Extracts features for classification ny frames for .wav file

        :param wav_path: string, path to .wav file
        :param data_path: string, path to all files and lables.txt
        :return: pandas.DataFrame with features of shape (n_chunks, n_features)
        """

        segments_number = 474
        y, sr = librosa.load(wav_path)

        melspec_features = 128
        mfcc_features = 13
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=melspec_features)
        log_S = librosa.power_to_db(S, ref=np.max)
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=mfcc_features)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        data_sampler = SSPNetDataSampler(data_path)

        if log_S.shape[1] != 474:
            return None
        labels = np.array(data_sampler.get_labels_for_file(wav_path, frame_sec=(11./segments_number))['IS_LAUGHTER'])
        labels = labels[:segments_number].reshape(segments_number, 1)

        M = np.concatenate((mfcc, delta_mfcc, delta2_mfcc, log_S)).T
        M = np.concatenate((M, labels), axis=1)

        col_names = ["{}".format(i) for i in range(melspec_features + 3 * mfcc_features)]
        col_names.append("IS_LAUGHTER")

        return pd.DataFrame(M, columns=col_names)


class PyAAExtractor(FeatureExtractor):
    """Python Audio Analysis features extractor"""
    def __init__(self):
        self.extract_script = "./extract_pyAA_features.py"
        self.py_env_name = "ipykernel_py2"

    def extract_features(self, wav_path):
        with tempfile.NamedTemporaryFile() as tmp_file:
            feature_save_path = tmp_file.name
            cmd = "python \"{}\" --wav_path=\"{}\" " \
                  "--feature_save_path=\"{}\"".format(self.extract_script, wav_path, feature_save_path)
            os.system("source activate {}; {}".format(self.py_env_name, cmd))

            feature_df = pd.read_csv(feature_save_path)
        return feature_df
