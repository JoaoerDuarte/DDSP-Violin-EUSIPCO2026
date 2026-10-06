import torch
import torch.nn as nn

from .core import exp_sigmoid, remove_above_nyquist, scale_db, scale_f0_hz
from .decoder import Decoder
from .encoder import Encoder
from .filters import body_filter
from .model_utils import (
    AMPS, BOW_POSITION_RAW, BRIGHTNESS_RAW, F0_SCALED, HARMONIC_DISTRIBUTION, LD_SCALED,
    NOISE_MAGNITUDES, NOTCH_DEPTH_RAW, RESIDUALS_RAW, Z, physics_harmonic_composer, scaled_sigmoid,
)
from .synth import filtered_noise_synth, harmonic_synth


class DDSP(nn.Module):
    """DDSP with an MFCC encoder, filtered noise and a learnable body filter (FIR by default,
    AR or ARMA with resonance_type).

    physics_source selects the harmonic source: the DDSP-Violin source (a 1/n distribution
    shaped by the bow, brightness and residual masks) or a harmonic distribution predicted
    freely by the decoder (the baseline). The other arguments default to the paper's settings
    and include its ablation switches.
    """

    def __init__(self, sampling_rate: int, block_size: int, signal_length: int, n_harmonic: int,
                 n_bands: int, encoder: dict, decoder: dict, physics_source: bool,
                 bow_mask: bool = True, brightness_mask: bool = True, residuals_mask: bool = True,
                 alpha_range: tuple = (-1.0, 1.0), beta_range: tuple = (0.05, 0.5),
                 n_residuals: int = 10, residual_scale: float = 1.0, smooth_bow_notch: bool = True,
                 notch_width: float = 0.05, brightness_tilt: bool = False,
                 init_harmonic_1_over_n: bool = False, noise_bias: float = -5.0,
                 resonance_type: str = "convolutional", resonance_length: int = 2048,
                 resonance_ar_order: int = 64, resonance_ma_order: int = 64,
                 resonance_window_size: int = 1600, resonance_max_reflection: float = 1.0,
                 resonance_position: str = "before_noise"):
        super().__init__()
        if resonance_position not in ("before_noise", "after_noise"):
            raise ValueError(f"resonance_position must be before_noise or after_noise, "
                             f"not {resonance_position!r}")
        self.physics_source = physics_source
        self.bow_mask = bow_mask
        self.brightness_mask = brightness_mask
        self.residuals_mask = residuals_mask
        self.brightness_tilt = brightness_tilt and not physics_source
        self.alpha_range = (float(alpha_range[0]), float(alpha_range[1]))
        self.beta_range = (float(beta_range[0]), float(beta_range[1]))
        self.n_residuals = n_residuals
        self.residual_scale = float(residual_scale)
        self.smooth_bow_notch = smooth_bow_notch
        self.notch_width = notch_width
        self.n_harmonic = n_harmonic
        self.resonance_position = resonance_position
        self.register_buffer("sampling_rate", torch.tensor(float(sampling_rate)))
        self.register_buffer("block_size", torch.tensor(int(block_size)))
        self.register_buffer("noise_bias", torch.tensor(float(noise_bias)))

        self.encoder = Encoder(sample_rate=sampling_rate, target_length=signal_length // block_size,
                               **encoder)

        outputs = [(AMPS, 1)]
        if physics_source:
            if bow_mask:
                outputs += [(BOW_POSITION_RAW, 1), (NOTCH_DEPTH_RAW, 1)]
            if brightness_mask:
                outputs.append((BRIGHTNESS_RAW, 1))
            if residuals_mask:
                outputs.append((RESIDUALS_RAW, n_residuals))
        else:
            outputs.append((HARMONIC_DISTRIBUTION, n_harmonic))
            if brightness_tilt:
                outputs.append((BRIGHTNESS_RAW, 1))
        outputs.append((NOISE_MAGNITUDES, n_bands))
        self.decoder = Decoder([F0_SCALED, LD_SCALED, Z], outputs, encoder["z_dims"], **decoder)
        if init_harmonic_1_over_n and not physics_source:
            self.decoder.init_harmonic_bias_1_over_n()

        self.resonance = body_filter(resonance_type, resonance_length, resonance_ar_order,
                                     resonance_ma_order, resonance_window_size, resonance_max_reflection)

    def forward(self, pitch: torch.Tensor, loudness: torch.Tensor, audio: torch.Tensor) -> dict:
        """pitch (Hz) and loudness (dB): [batch, frames, 1]; audio: [batch, samples].

        Returns the output 'signal' [batch, samples, 1] and the source controls: 'alpha',
        'beta', 'gamma' and 'residuals' (DDSP-Violin) or 'harmonic_amplitudes' (baseline).
        """
        sampling_rate = int(self.sampling_rate.item())
        block_size = int(self.block_size.item())
        params = self.decoder(**{F0_SCALED: scale_f0_hz(pitch), LD_SCALED: scale_db(loudness),
                                 Z: self.encoder(audio)})
        total_amplitude = exp_sigmoid(params[AMPS])
        noise_magnitudes = exp_sigmoid(params[NOISE_MAGNITUDES] + self.noise_bias, threshold=0.0)

        if self.physics_source:
            outputs = self._source_controls(params, pitch)
            amplitudes = physics_harmonic_composer(
                outputs['beta'], outputs['gamma'], outputs['alpha'], outputs['residuals'], pitch,
                self.n_harmonic, float(sampling_rate), self.bow_mask, self.brightness_mask,
                self.residuals_mask, self.smooth_bow_notch, self.notch_width,
            )
        else:
            amplitudes = exp_sigmoid(params[HARMONIC_DISTRIBUTION])
            if self.brightness_tilt:
                alpha = scaled_sigmoid(params[BRIGHTNESS_RAW], *self.alpha_range)
                n = torch.arange(1, self.n_harmonic + 1, device=amplitudes.device, dtype=amplitudes.dtype)
                amplitudes = amplitudes * (n.view(1, 1, -1) ** (-alpha))
            amplitudes = remove_above_nyquist(amplitudes, pitch, float(sampling_rate))
            amplitudes = amplitudes / (amplitudes.sum(-1, keepdim=True) + 1e-7)
            outputs = {'harmonic_amplitudes': amplitudes}

        signal = harmonic_synth(pitch, amplitudes, total_amplitude, sampling_rate, block_size)
        if self.resonance_position == "before_noise":
            signal = self.resonance(signal)
        signal = signal + filtered_noise_synth(noise_magnitudes, block_size)
        if self.resonance_position == "after_noise":
            signal = self.resonance(signal)
        outputs['signal'] = signal
        return outputs

    def _source_controls(self, params: dict, pitch: torch.Tensor) -> dict:
        """DDSP-Violin controls from the raw decoder outputs. A mask that is switched off
        gets neutral values (zeros for beta, gamma, alpha and ones for the residuals)."""
        batch_size, n_frames = pitch.shape[:2]
        if self.bow_mask:
            beta = scaled_sigmoid(params[BOW_POSITION_RAW], *self.beta_range)
            gamma = torch.sigmoid(params[NOTCH_DEPTH_RAW])
        else:
            beta = pitch.new_zeros(batch_size, n_frames, 1)
            gamma = pitch.new_zeros(batch_size, n_frames, 1)
        if self.brightness_mask:
            alpha = scaled_sigmoid(params[BRIGHTNESS_RAW], *self.alpha_range)
        else:
            alpha = pitch.new_zeros(batch_size, n_frames, 1)
        if self.residuals_mask:
            residuals = 1.0 + self.residual_scale * torch.tanh(params[RESIDUALS_RAW])
        else:
            residuals = pitch.new_ones(batch_size, n_frames, self.n_residuals)
        return {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'residuals': residuals}
