"""Preprocess the training audio: segments of model.signal_length samples with their f0
(CREPE) and A-weighted loudness per frame of model.block_size samples."""
import argparse
import os
import pathlib

import librosa
import numpy as np
import torch
import yaml
from tqdm import tqdm

from ddsp_torch.core import extract_loudness, extract_pitch

ARRAYS = ("signals", "pitches", "loudness")


def load_config(name: str) -> dict:
    """Load configs/<name>.yaml."""
    with open(os.path.join("configs", f"{name}.yaml")) as f:
        return yaml.safe_load(f)


def data_dirs(config: dict) -> tuple[str, str]:
    """Folder of the training .wav files and folder of their preprocessed arrays."""
    audio_dir = config.get("data", {}).get("audio_dir", "dataset/audio")
    return audio_dir, os.path.join("preprocessed", os.path.basename(os.path.normpath(audio_dir)))


def preprocess_file(path, sampling_rate: int, block_size: int, signal_length: int):
    """Resample one file and zero-pad it to whole segments. Returns the segments
    [n, signal_length] with their f0 (Hz) and loudness (dB) [n, signal_length // block_size]."""
    audio, _ = librosa.load(path, sr=sampling_rate)
    audio = np.pad(audio, (0, -len(audio) % signal_length))
    pitch = extract_pitch(audio, sampling_rate, block_size)
    loudness = extract_loudness(audio, sampling_rate, block_size)
    n_segments = len(audio) // signal_length
    n_frames = n_segments * (signal_length // block_size)
    return (audio.reshape(n_segments, signal_length),
            pitch[:n_frames].reshape(n_segments, -1),
            loudness[:n_frames].reshape(n_segments, -1))


def preprocess(config: dict):
    """Preprocess every .wav file under data.audio_dir and save the arrays."""
    model = config["model"]
    audio_dir, out_dir = data_dirs(config)
    files = sorted(pathlib.Path(audio_dir).rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No .wav files in {audio_dir}")
    results = [preprocess_file(f, model["sampling_rate"], model["block_size"], model["signal_length"])
               for f in tqdm(files, desc="Preprocessing")]
    os.makedirs(out_dir, exist_ok=True)
    for name, arrays in zip(ARRAYS, zip(*results)):
        np.save(os.path.join(out_dir, f"{name}.npy"), np.concatenate(arrays).astype(np.float32))
    print(f"Saved {sum(len(r[0]) for r in results)} segments to {out_dir}")


class Dataset(torch.utils.data.Dataset):
    """Preprocessed segments as (audio, f0, loudness)."""

    def __init__(self, data_dir: str):
        self.signals, self.pitches, self.loudness = (
            np.load(os.path.join(data_dir, f"{name}.npy")) for name in ARRAYS)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, i):
        return (torch.from_numpy(self.signals[i]), torch.from_numpy(self.pitches[i]),
                torch.from_numpy(self.loudness[i]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ddsp_violin", help="name of a config in configs/")
    preprocess(load_config(parser.parse_args().config))
