import soundfile
import numpy as np
import random
import argparse
from pathlib import Path


def random_noise_file():
    return str(random.choice(list(Path("./bg_noise").rglob("*.wav")) + list(list(Path("./bg_noise").rglob("*.flac")))))


# For different samplerates
def scale(data, target_samplerate, current_samplerate):
    result = []
    p_v = 0
    counter = 0
    # k + counter * target_samplerate / current_samplerate >= (counter + 1) * target_samplerate / current_samplerate
    for v in data:
        k = 0
        while k + counter * target_samplerate // current_samplerate < (counter + 1) * target_samplerate // current_samplerate:
            k += 1
        for i in range(1, k+1):
            result.append(p_v + (p_v - v) * i / k)
        p_v = v
        counter += 1
    return np.array(result)


def cut(data, length, random_start=False):
    res = np.array([])
    if random_start:
        res = np.append(res, data[np.random.randint(0, len(data) - 1):])
    while len(res) < length:
        res = np.append(res, data)
    return res[:length]


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_file", type=str, default="input.wav")
    arg_parser.add_argument("output_file", type=str, default="output.wav")
    args = arg_parser.parse_args()

    input_filename = args.input_file
    output_filename = args.output_file
    noise_power = np.random.uniform(0.05, 0.2)

    input_data, input_samplerate = soundfile.read(input_filename)
    noise_data, noise_samplerate = soundfile.read(random_noise_file())

    samplerate = max(input_samplerate, noise_samplerate)
    input_data = scale(input_data, samplerate, input_samplerate)
    noise_data = scale(noise_data, samplerate, noise_samplerate)

    result = input_data + noise_power * cut(noise_data, len(input_data), True)

    soundfile.write(output_filename, result, samplerate)
