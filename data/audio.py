import librosa


def load_audio(file_path, sample_rate, start=0, end=None):
    if end is None:
        duration = None
    else:
        duration = end - start

    audio, _ = librosa.load(file_path, sr=sample_rate, offset=start, duration=duration)
    return audio
