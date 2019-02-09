## HW:Laughter detector

You need to automatically detect laughter in audio file.

You can use the audio corpus available at http://www.dcs.gla.ac.uk/vincia/?p=378 (vocalizationcorpus.zip) as train / test dataset.
To save time you will be provided with open implementation which you're free to modify

The preparation of train / test datasets are full responsibility of a student.

Task:
*  Learn how to extract MFCC + FBANK features using librosa and add new feature extractor in laughter_prediction.feature_extractors.py
*  Implement a new predictor in lauhter_prediction.predictors.py based on RNN architecture. Use separate loss function for MFCC features only and for the final output. Example can be found here: https://keras.io/getting-started/functional-api-guide/ at Multi-input and multi-output models section
*  Add noise to dataset using robustiier from the first homework or download dataset with added noise from https://yadi.sk/d/zRHkhtE83T4kV3 and compare prediction accuracy and clean data and data with added noise. Which dataset you'd use for train and which for test?
*  Publish notebook with framewise AUC and plot a couple of audio samples with true laughter intervals with predicted laughter probability

Requirements:
* Laughter detector:
    * Input: audio file
    * Output: detected laughter segments
* It’s necessary to present the description of algorithm and detector statistics (accuracy, ...).
* The data used for training / testing should be available, so that it will be possible to train the classifier from scratch.
* All materials (source codes, description, …) should be committed to github.
