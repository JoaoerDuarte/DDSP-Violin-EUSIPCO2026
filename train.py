"""Train DDSP-Violin (default) or the baseline, e.g. python train.py --config baseline."""
import argparse
import os
import subprocess
import sys

import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ddsp_torch.losses import HarmonicResidualLoss, MultiScaleSTFTLoss
from ddsp_torch.model import DDSP
from preprocess import ARRAYS, Dataset, data_dirs, load_config

# Training settings of the paper. A config can override them under "train".
TRAIN_DEFAULTS = {
    "steps": 10000,
    "batch_size": 8,
    "learning_rate": 1e-3,
    "lr_decay_rate": 0.98,  # learning-rate factor per lr_decay_steps, spread over every step
    "lr_decay_steps": 10000,
    "gradient_clip_norm": 3.0,
    "num_workers": 4,
}


def build_losses(config: dict):
    """Multi-scale STFT loss, and the Harmonic Residual Loss when loss.hrl_weight > 0
    (DDSP-Violin source only). Returns (mssl, hrl), hrl being None when off."""
    loss = config["loss"]
    hrl_weight = loss.get("hrl_weight", 0.0)
    use_hrl = config["model"]["physics_source"] and hrl_weight > 0
    return MultiScaleSTFTLoss(**loss["mssl"]), HarmonicResidualLoss(hrl_weight) if use_hrl else None


def compute_loss(mssl, hrl, outputs: dict, audio, pitch, loudness):
    """Returns the total loss, the spectral loss and the HRL term (None without HRL)."""
    spectral = mssl(outputs["signal"].squeeze(-1), audio)
    if hrl is None:
        return spectral, spectral, None
    hrl_term = hrl(outputs["residuals"], pitch, loudness)
    return spectral + hrl_term, spectral, hrl_term


def train(config_name: str):
    config = load_config(config_name)
    settings = {**TRAIN_DEFAULTS, **config.get("train", {})}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, data_dir = data_dirs(config)
    if not all(os.path.exists(os.path.join(data_dir, f"{name}.npy")) for name in ARRAYS):
        # Separate process, so that TensorFlow (used by CREPE) releases the GPU memory before training.
        subprocess.run([sys.executable, "preprocess.py", "--config", config_name], check=True)
    loader = torch.utils.data.DataLoader(
        Dataset(data_dir), settings["batch_size"], shuffle=True, drop_last=True,
        num_workers=settings["num_workers"], pin_memory=device.type == "cuda",
        persistent_workers=settings["num_workers"] > 0,
    )
    if len(loader) == 0:
        raise ValueError(f"{data_dir} holds fewer segments than one batch")

    model = DDSP(**config["model"]).to(device)
    mssl, hrl = build_losses(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=settings["lr_decay_rate"] ** (1 / settings["lr_decay_steps"]))

    run_dir = os.path.join("runs", config_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)
    writer = SummaryWriter(run_dir)

    step = 0
    with tqdm(total=settings["steps"], desc="Training") as progress:
        while step < settings["steps"]:
            for audio, pitch, loudness in loader:
                audio = audio.to(device)
                pitch = pitch.unsqueeze(-1).to(device)
                loudness = loudness.unsqueeze(-1).to(device)
                total, spectral, hrl_term = compute_loss(
                    mssl, hrl, model(pitch, loudness, audio), audio, pitch, loudness)
                optimizer.zero_grad()
                total.backward()
                clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"])
                optimizer.step()
                scheduler.step()

                writer.add_scalar("Loss/Total", total.item(), step)
                writer.add_scalar("Loss/Spectral", spectral.item(), step)
                if hrl_term is not None:
                    writer.add_scalar("Loss/HRL", hrl_term.item(), step)
                progress.set_postfix(loss=f"{total.item():.4f}")
                progress.update()
                step += 1
                if step == settings["steps"]:
                    break
    writer.close()
    torch.save(model.state_dict(), os.path.join(run_dir, "final_state.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ddsp_violin", help="name of a config in configs/")
    train(parser.parse_args().config)
