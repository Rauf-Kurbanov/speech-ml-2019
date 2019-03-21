import argparse
import json


def intervals_gen(timestamps, frame_sec, error_dist=None, min_frames=None):
    """
    Generator returning target class intervals from timestamps

    :param timestamps: array-like, starting point of each frame in sec
    :param frame_sec: int, length of each timeframe in seconds
    :param error_dist: error_dist: int, allowed length of misclassified audio frame in sec
    :param min_frames: int, minimal valid interval length in sec
    :return: generator returning pairs: (interval_start, interval_end)
    """
    if min_frames is None:
        min_frames = 10
    if error_dist is None:
        error_dist = 1.5 * frame_sec

    begin = 0
    length = len(timestamps)
    if length <= 1:
        return
    for i in range(1, length):
        if timestamps[i] - timestamps[i - 1] >= error_dist:
            if i - 1 - begin > min_frames:
                yield timestamps[begin], timestamps[i - 1]
            begin = i
    if begin != length - 1 and length - 1 - begin > min_frames:
        yield timestamps[begin], timestamps[length - 1]


def predicted_to_intervals(pred_classes, frame_sec, error_dist=None):
    """
    Extracts target class intervals from binary predictions by frames

    :param pred_classes: array-like, binary prediction for each timeframe
    :param frame_sec: int, length of each timeframe in seconds
    :param error_dist: int, allowed length of misclassified audio frame in sec
    :return: array of pairs, valid target class intervals
    """
    if error_dist is None:
        error_dist = frame_sec * 1.5
    frames_to_times = [frame_sec * i for i, pred in enumerate(pred_classes) if pred == 1]
    intervals_g = intervals_gen(frames_to_times, frame_sec=frame_sec, error_dist=error_dist)
    intervals = list(intervals_g)
    return intervals


def my_import(name):
    """
    Imports class for any module by name

    :param name: string, dot-separated full class name
    :return: callable, returning class instance
    """

    # http://stackoverflow.com/questions/547829/how-to-dynamically-load-a-python-class
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def main():
    DEFALUT_PARAMS = "params/default_params.json"

    parser = argparse.ArgumentParser(description='Script for prediction laughter intervals for .wav file')
    parser.add_argument('--wav_path', type=str, help='Path to .wav file')
    parser.add_argument('--params', type=str, default=DEFALUT_PARAMS,
                        help='/JSON file with the classification parameters. Default: ' + DEFALUT_PARAMS + '.')
    args = parser.parse_args()

    with open(args.params, 'r') as params_file:
        params = json.load(params_file)

    klass = my_import(params['predictor'])
    predictor = klass()
    klass = my_import(params['extractor'])
    extractor = klass()

    feature_df = extractor.extract_features(args.wav_path)
    pred_classes = predictor.predict(feature_df.as_matrix())
    intervals = predicted_to_intervals(pred_classes, frame_sec=0.01, error_dist=0.1)
    print("Target intervals")
    print(intervals)


if __name__ == '__main__':
    main()
