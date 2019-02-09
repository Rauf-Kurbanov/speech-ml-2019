import re


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def in_any(x, ranges):
    return any([x in rr for rr in ranges])


def time_to_num(time, sample_len, duration):
    return int(sample_len * time / duration)


def interv_to_range(interv, slen, duration):
    fr, to = time_to_num(interv[0], slen, duration), time_to_num(interv[1], slen, duration)
    return range(fr, to)


def get_sname(wav_path):
    return re.search('(S[0-9]*).wav', wav_path).group(1)
