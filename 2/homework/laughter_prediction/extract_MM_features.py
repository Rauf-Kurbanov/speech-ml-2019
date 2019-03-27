from __future__ import print_function

import os
import argparse
from homework.laughter_prediction.feature_extractors import FeatureExtractor


def main():
    parser = argparse.ArgumentParser(description='Feature extraction script based on MFCC and melspectrogram features')
    parser.add_argument('--wav_path', type=str, help='Path to directory with .wav files')
    parser.add_argument('--labels_path', type=str, help='Path to labels file')
    parser.add_argument('--feature_save_path', type=str, help='Path to save features .csv file')
    parser.add_argument('--segments_count', type=str, help='Number of segments that cut .wav file')
    args = parser.parse_args()

    feature_extractor = FeatureExtractor()

    wav_files = os.listdir(args.wav_path)
    if args.segments_count == 11 or args.segments_count is None:
        for wav in wav_files:
            print(wav)
            try:
                df = feature_extractor.extract_features(wav_path=args.wav_path + wav, data_path=args.labels_path)
            except Exception:
                continue

            if df.shape[0] == 11:

                df.to_csv(args.feature_save_path, mode='a', header=False, index=False)
    elif args.segments_count == 474:
        for wav in wav_files:
            print(wav)
            try:
                df = feature_extractor.extract_features474(wav_path=args.wav_path + wav, data_path=args.labels_path)
            except Exception:
                continue

            if df.shape[0] == 11:
                df.to_csv(args.feature_save_path, mode='a', header=False, index=False)

    print("Done")


if __name__ == '__main__':
    main()
