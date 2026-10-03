"""Names of the decoder inputs and outputs, and the DDSP-Violin harmonic source."""
import torch

F0_SCALED = 'f0_scaled'
LD_SCALED = 'ld_scaled'
Z = 'z'
AMPS = 'amps'
HARMONIC_DISTRIBUTION = 'harmonic_distribution'
NOISE_MAGNITUDES = 'noise_magnitudes'
BOW_POSITION_RAW = 'bow_position_raw'
NOTCH_DEPTH_RAW = 'notch_depth_raw'
BRIGHTNESS_RAW = 'brightness_raw'
RESIDUALS_RAW = 'residuals_raw'


def scaled_sigmoid(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Map raw decoder outputs to [low, high] with a sigmoid."""
    return low + (high - low) * torch.sigmoid(raw)


def compute_bow_notch(β: torch.Tensor, n_harmonic: int) -> torch.Tensor:
    """Bow-position notch pattern sin²(π n β) for harmonics n = 1..n_harmonic."""
    batch_size, time_steps, _ = β.shape
    n = torch.arange(1, n_harmonic + 1, device=β.device, dtype=β.dtype).view(1, 1, -1)
    n = n.expand(batch_size, time_steps, -1)
    return torch.sin(torch.pi * n * β.expand(-1, -1, n_harmonic)) ** 2


def smooth_notch(notch: torch.Tensor, width: float, n_harmonic: int) -> torch.Tensor:
    """Smooth the notch pattern along the harmonic axis: zero-pad it to 2 * n_harmonic,
    multiply its rfft by a Gaussian of standard deviation width / 2π and invert."""
    batch_size, time_steps, _ = notch.shape
    n_fft = 2 * n_harmonic
    notch_fft = torch.fft.rfft(torch.nn.functional.pad(notch, (0, n_fft - n_harmonic)), dim=-1)
    freqs = torch.linspace(0, 0.5, notch_fft.shape[-1], device=notch.device, dtype=notch.dtype)
    sigma = width / (2.0 * torch.pi)
    window = torch.exp(-0.5 * (freqs / sigma) ** 2)
    window = window.view(1, 1, -1).expand(batch_size, time_steps, -1)
    return torch.fft.irfft(notch_fft * window, n=n_fft, dim=-1)[:, :, :n_harmonic]


def physics_harmonic_composer(β: torch.Tensor, γ: torch.Tensor, α: torch.Tensor, residuals: torch.Tensor,
                              pitch: torch.Tensor, n_harmonic: int, sampling_rate: float,
                              use_bow_mask: bool = True, use_brightness_mask: bool = True,
                              use_residuals_mask: bool = True, smooth_bow_notch: bool = True,
                              notch_width: float = 0.05) -> torch.Tensor:
    """Harmonic distribution c_n proportional to (1/n) * bow mask * brightness mask * residuals,
    with harmonics above Nyquist removed and normalised over n. A frame whose harmonics all lie
    above Nyquist puts its whole weight on the fundamental.

    β, γ, α, pitch (Hz): [batch, frames, 1]; residuals: [batch, frames, n_residuals]
    for the first n_residuals harmonics. Returns [batch, frames, n_harmonic].
    """
    batch_size, time_steps, _ = β.shape
    n = torch.arange(1, n_harmonic + 1, device=β.device, dtype=β.dtype).view(1, 1, -1)
    n = n.expand(batch_size, time_steps, -1)
    spectrum = 1.0 / n
    if use_bow_mask:
        notch = compute_bow_notch(β, n_harmonic)
        if smooth_bow_notch:
            notch = smooth_notch(notch, notch_width, n_harmonic)
        spectrum = spectrum * (1.0 - γ * (1.0 - notch))
    if use_brightness_mask:
        spectrum = spectrum * n ** (-α)
    if use_residuals_mask:
        residuals_full = torch.ones(batch_size, time_steps, n_harmonic, device=β.device, dtype=β.dtype)
        residuals_full[:, :, :residuals.shape[-1]] = residuals
        spectrum = spectrum * residuals_full
    spectrum = spectrum * (pitch * n < sampling_rate / 2.0).to(β.dtype)
    spectrum_sum = spectrum.sum(dim=-1, keepdim=True)
    safe = (spectrum_sum > 1e-7).to(β.dtype)
    fallback = torch.zeros_like(spectrum)
    fallback[:, :, 0] = 1.0
    return torch.where(safe.expand_as(spectrum) > 0.5, spectrum / (spectrum_sum + 1e-7), fallback)
