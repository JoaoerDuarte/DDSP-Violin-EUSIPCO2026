# DDSP-Violin

Code accompanying **"DDSP-Violin: Physically-Informed Constraints for Disentangled Source-Filter Decomposition"** (EUSIPCO 2026).

A physically-informed DDSP framework for bowed-string synthesis. Constrains the harmonic source with low-dimensional parameters inspired by bowed-string acoustics (brightness $\alpha$, bow position $\beta$, notch depth $\gamma$, residuals $\rho_n$) to improve source-filter disentanglement.

## Audio examples

Listen at [joaoerduarte.github.io/DDSP-Violin-EUSIPCO2026](https://joaoerduarte.github.io/DDSP-Violin-EUSIPCO2026/).

## Setup

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py                     # DDSP-Violin, configs/ddsp_violin.yaml
python train.py --config baseline   # Baseline, configs/baseline.yaml
```

Data: `.wav` files in `dataset/audio/` (or the folder set by `data: {audio_dir: ...}` in the config), resampled to 16 kHz. The paper trains one model per body impulse response. The first run extracts f0 (CREPE) and loudness into `preprocessed/`, and `python preprocess.py` runs this step alone. Weights are saved to `runs/<config>/final_state.pth`.

## Evaluation

`python evaluate.py --responses <folder> runs/<config> ...` prints the body-filter errors of the paper (MC-LSD, tilt and resonance error) against measured body responses, which are not included here. Each run is compared with the `.wav` response whose name appears in its folder name.

## Citation

```bibtex
@inproceedings{duarte2026ddspviolin,
  title={{DDSP-Violin}: Physically-Informed Constraints for Disentangled Source-Filter Decomposition},
  author={Duarte, Jo{\~a}o and Mignot, R{\'e}mi and McDermott, James and O'Leary, Se{\'a}n},
  booktitle={Proc. EUSIPCO},
  year={2026}
}
```

## Acknowledgments

Built upon [acids-ircam/ddsp_pytorch](https://github.com/acids-ircam/ddsp_pytorch). Original DDSP framework: Engel et al. (ICLR 2020). Bowed-string physical model: Demoucron (2008).

Supported by the Research Ireland Centre for Research Training in Digitally-Enhanced Reality (d-real), Grant No. 18/CRT/6224.

## License

MIT
