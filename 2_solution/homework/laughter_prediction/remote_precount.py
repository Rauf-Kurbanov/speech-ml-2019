from __future__ import print_function

import os
import argparse
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


def main():
    parser = argparse.ArgumentParser(description='Feature extraction script based on MFCC and melspectrogram features')
    parser.add_argument('--wav_path', type=str, help='Path to directory with .wav files')
    parser.add_argument('--labels_path', type=str, help='Path to labels file')
    parser.add_argument('--feature_save_path', type=str, help='Path to save features .csv file')
    args = parser.parse_args()

    feature_extractor = FeatureExtractor()

    wav_files = os.listdir("../../data/")
    for wav in wav_files:
        print(wav)
        try:
            df = feature_extractor.extract_features(wav_path="../../data/" + wav, data_path="../../")
        except Exception:
            continue

        if df.shape[0] == 11:

            df.to_csv("../../super_mega_file.csv", mode='a', header=False, index=False)

    print("Done")


if __name__ == '__main__':
    main()
