## Laughter detection tool

This tool detects laughter interval in audio files.

### Repo structure
laughter_classification - everything related with frame-wise 
 laughter classification: model training, visualization, tuning,
  cross-valiadation

laughter_classification_test - test for helper classes
 
laughter_prediction - module for laughter preidiction for
arbitrary audio file in .wav format

models - serialized pre-learned models for classifiation

features - pre-extracted features in .csv format

params - configuration files for laughter prediction

### Data
Audio corpus available at 
http://www.dcs.gla.ac.uk/vincia/?p=378 (vocalizationcorpus.zip)

### Serialized models
Available at https://goo.gl/D04Wmm

### Pre-extracted features
Available at https://goo.gl/CkRjZO

### How to use
usage: process_audio.py [-h] [--wav_path WAV_PATH] [--params PARAMS]

Script for prediction laughter intervals for .wav file

optional arguments:
  -h, --help           show this help message and exit
  --wav_path WAV_PATH  Path to .wav file
  --params PARAMS      /JSON file with the classification parameters. Default:
                       ../params/default_params.json.

### Note
Tool expects you to have python2 kernel with installed pyAudioAnalysis
named ipykernel_py2
