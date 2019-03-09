## HW:Noisifier

Implement a script (preferably python) for duplicating audio data directori and adding background noise.

You can use the noise available at https://yadi.sk/d/ZR5JdkhO3SPoLN (bg_noise.tar.gz)

     ================= NOISE TO BE ADDED DATA ================ #
     noise_root = 'bg_noise/'
     structure:
       / noise_dataset
               / arbitrary contents, wavs somewhere
     Noise_datasets:
       'tut_bg_noise_corpus_gsm' (background nosies),
       'AUDIONAUTIX_MUSIC_gsm' (cc music tracks),
       'ASTERISK_MUSIC_gsm' (phone on hold music track)
       'FRESOUND_BEEPS_gsm' (beeping sounds from freesound, cc)
       'freesound_foreground_gsm (extra noises from freesound)'
       'freesound_background_gsm (extra background noises)'
    
Requirements:
* Support .wav and .flac formats
* Use both beeps and backgrownd music from bg_noise.tar.gz
* add an example audio files
* add HOWTO into README
