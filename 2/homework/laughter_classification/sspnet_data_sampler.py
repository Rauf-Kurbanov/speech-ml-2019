import os
from os.path import join

import numpy as np
import pandas as pd
import scipy.io.wavfile as wav

from laughter_classification.utils import chunks, in_any, interv_to_range, get_sname

from laughter_prediction.sample_audio import sample_wav_by_time


class SSPNetDataSampler:
    """
    Class for loading and sampling audio data by frames for SSPNet Vocalization Corpus
    """

    @staticmethod
    def read_labels(labels_path):
        def_cols = ['Sample', 'original_spk', 'gender', 'original_time']
        label_cols = ["{}_{}".format(name, ind) for ind in range(6) for name in ('type_voc', 'start_voc', 'end_voc')]
        def_cols.extend(label_cols)
        labels = pd.read_csv(labels_path, names=def_cols, engine='python', skiprows=1)
        return labels

    def __init__(self, corpus_root):
        self.sample_rate = 16000
        self.duration = 11
        self.default_len = self.sample_rate * self.duration
        self.data_dir = join(corpus_root, "data")
        labels_path = join(corpus_root, "labels.txt")
        self.labels = self.read_labels(labels_path)

    @staticmethod
    def most(l):
        return int(sum(l) > len(l) / 2)

    @staticmethod
    def _interval_generator(incidents):
        for itype, start, end in chunks(incidents, 3):
            if itype == 'laughter':
                yield start, end

    def get_labels_for_file(self, wav_path, frame_sec):
        sname = get_sname(wav_path)
        sample = self.labels[self.labels.Sample == sname]

        incidents = sample.loc[:, 'type_voc_0':'end_voc_5']
        incidents = incidents.dropna(axis=1, how='all')
        incidents = incidents.values[0]

        rate, audio = wav.read(wav_path)

        laughts = self._interval_generator(incidents)
        laughts = [interv_to_range(x, len(audio), self.duration) for x in laughts]
        laught_along = [1 if in_any(t, laughts) else 0 for t, _ in enumerate(audio)]

        frame_size = int(self.sample_rate * frame_sec)
        is_laughter = np.array([self.most(la) for la in chunks(laught_along, frame_size)])

        df = pd.DataFrame({'IS_LAUGHTER': is_laughter,
                           'SNAME': sname})
        return df

    def df_from_file(self, wav_path, frame_sec):
        """
        Returns sampled data by path to audio file
        :param wav_path: string, .wav file path
        :param frame_sec: int, length of each frame in sec
        :return: pandas.DataFrame with sampled audio
        """
        data = sample_wav_by_time(wav_path, frame_sec)
        labels = self.get_labels_for_file(wav_path, frame_sec)
        df = pd.concat([data, labels], axis=1)
        return df

    def get_valid_wav_paths(self):
        for dirpath, dirnames, filenames in os.walk(self.data_dir):
            fullpaths = [join(dirpath, fn) for fn in filenames]
            return [path for path in fullpaths if len(wav.read(path)[1]) == self.default_len]

    def create_sampled_df(self, frame_sec, naudio=None, save_path=None, force_save=False):
        """
        Returns sampled data for whole corpus
        :param frame_sec: int, length of each frame in sec
        :param naudio: int, number of audios to parse, if not defined parses all
        :param save_path: string, path to save parsed corpus
        :param force_save: boolean, if you want to override file with same name
        :return:
        """
        fullpaths = self.get_valid_wav_paths()[:naudio]
        dataframes = [self.df_from_file(wav_path, frame_sec) for wav_path in fullpaths]
        df = pd.concat(dataframes)

        colnames = ["V{}".format(i) for i in range(df.shape[1] - 2)]
        colnames.append("IS_LAUGHTER")
        colnames.append("SNAME")
        df.columns = colnames

        if save_path is not None:
            if not os.path.isfile(save_path) or force_save:
                print("saving df: ", save_path)
                df.to_csv(save_path, index=False)

        return df
