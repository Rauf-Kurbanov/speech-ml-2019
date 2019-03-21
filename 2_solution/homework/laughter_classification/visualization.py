import os
from os.path import join

import librosa
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc


def plot_corr_mat(df):
    sns.set(context="paper", font="monospace")
    corrmat = df.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    f.tight_layout()


def plot_sequence(seq):
    x = range(len(seq))
    plt.figure(figsize=(80, 10))
    plt.plot(x, seq)
    plt.show()


def plot_ROC_curve(y_score, y_expected):
    fpr, tpr, thresholds = roc_curve(y_expected, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


class WavVisualizer:

    def __init__(self, corpus_root, sample_rate, labels):
        self.SAMPLE_RATE = sample_rate
        self.CORPUS_ROOT = corpus_root
        self.DATA_DIR = join(corpus_root, "data")
        self.DURATION = 11
        self.labels = labels

    def plot_audio(self, audio_path):
        sr = self.SAMPLE_RATE
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        x = range(len(audio))
        plt.figure(figsize=(80, 10))
        plt.plot(x, audio)
        plt.show()

    def _time_to_num(self, time, sample_len):
        duration = self.DURATION
        return int(sample_len * time / duration)

    @staticmethod
    def _in_any(x, ranges):
        return any([x in rr for rr in ranges])

    def interv_to_range(self, interv, slen):
        _time_to_num = self._time_to_num
        fr, to = _time_to_num(interv[0], slen), _time_to_num(interv[1], slen)
        return range(fr, to)

    def show(self, audio_path, laughts):
        audio, _ = librosa.load(audio_path, sr=self.SAMPLE_RATE, mono=True)
        laughts = [self.interv_to_range(x, len(audio)) for x in laughts]

        x = range(len(audio))
        in_any = WavVisualizer._in_any
        no_laught_X = [t for t in x if not in_any(t, laughts)]
        no_laught_Y = [a for t, a in enumerate(audio) if not in_any(t, laughts)]
        laught_X = [t for t in x if in_any(t, laughts)]
        laught_Y = [a for t, a in enumerate(audio) if in_any(t, laughts)]

        plt.figure(figsize=(80, 10))
        plt.plot(no_laught_X, no_laught_Y, color='b')
        plt.plot(laught_X, laught_Y, color='r')
        plt.show()

    @staticmethod
    def _chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    @staticmethod
    def _interval_generator(incidents):
        chunks = WavVisualizer._chunks
        for itype, start, end in chunks(incidents, 3):
            if itype == 'laughter':
                    yield start, end

    def draw_sample(self, sname):
        data_dir = self.DATA_DIR
        labels = self.labels
        sample = labels[labels.Sample == sname]
        incidents = sample.loc[:, 'type_voc_0':'end_voc_5']
        incidents = incidents.iloc[0]
        incidents = [i for i in incidents if not pd.isnull(i)]

        wav_path = join(data_dir, sname + ".wav")
        laughts = WavVisualizer._interval_generator(incidents)
        self.show(wav_path, laughts)


if __name__ == '__main__':

    CORPUS_ROOT = "/media/rauf/TOSHIBA EXT1/STORAGE/DATA/vocalizationcorpus"
    DATA_DIR = join(CORPUS_ROOT, "data")
    SAMPLE_RATE = 16000
    DURATION = 11
    FRAME_SIZE = 100

    any_wav = os.listdir(DATA_DIR)[0]
    any_wav_path = join(DATA_DIR, any_wav)
    labels_path = join(CORPUS_ROOT, "labels.txt")

    def_cols = ['Sample', 'original_spk', 'gender', 'original_time']
    label_cols = ["{}_{}".format(name, ind) for ind in range(6) for name in ('type_voc', 'start_voc', 'end_voc')]
    def_cols.extend(label_cols)

    labels = pd.read_csv(labels_path, names=def_cols, engine='python', skiprows=1)

    wv = WavVisualizer(corpus_root=CORPUS_ROOT, sample_rate=SAMPLE_RATE, labels=labels)
    wv.plot_audio(any_wav_path)
