from __future__ import annotations
import numpy as np
import scipy.signal as signal
from config import AirBuddsConfig, DEFAULT_CONFIG

class PreambleGenerator:
    """
    LFM (Linear Frequency Modulation) Chirp Preamble Generator.
    Used for packet detection, coarse timing synchronization.
    """
    def __init__(self, config=None, sample_rate: int = 44100, fs: int = None):
        """
        Initialize the Preamble Generator.
        """
        self.config = config if config else DEFAULT_CONFIG.preamble
        self.sample_rate = fs if fs is not None else sample_rate
        
        self.f_start = getattr(self.config, 'f_start', 1000)
        self.f_end = getattr(self.config, 'f_stop', getattr(self.config, 'f_end', 10000))
        self.duration = getattr(self.config, 'duration', 0.05)
        self.num_repeats = getattr(self.config, 'num_repeats', 2)
        self.gap_duration = getattr(self.config, 'gap_duration', 0.01)
        self.length = getattr(self.config, 'length', None)
        
        # Precompute the basic chirp and full preamble
        self.chirp = self.generate_chirp()
        self.preamble = self.generate_preamble()

    def generate(self) -> np.ndarray:
        """Generate preamble signal."""
        if self.num_repeats > 1:
            return self.preamble
        return self.chirp

    def generate_chirp(self) -> np.ndarray:
        """
        Generate a single LFM chirp signal with a Hann window applied.
        s(t) = w(t) * cos(2π(f₀t + (B/2T)t²))
        """
        if self.length is not None and self.length > 0:
            length = self.length
            t = np.linspace(0, self.duration, length, endpoint=False)
        else:
            length = int(self.duration * self.sample_rate)
            t = np.arange(0, self.duration, 1.0 / self.sample_rate)[:length]
            
        if getattr(self.config, 'use_noise_preamble', False):
            # Deterministic pseudo-random noise sequence for stealthy synchronization
            np.random.seed(42)
            noise = np.random.normal(0, 1, length)
            
            # Bandpass filter the noise to match the chirp's frequency band
            nyq = self.sample_rate / 2
            low = max(self.f_start / nyq, 0.01)
            high = min(self.f_end / nyq, 0.99)
            if low < high:
                b, a = signal.butter(5, [low, high], btype='band')
                noise = signal.filtfilt(b, a, noise)
                
            w = np.hanning(len(noise))
            return (noise / np.max(np.abs(noise))) * w
            
        c = signal.chirp(t, f0=self.f_start, f1=self.f_end, t1=self.duration, method='linear')
        w = np.hanning(len(c))
        return c * w

    def generate_preamble(self) -> np.ndarray:
        """
        Generate the full preamble consisting of repeated chirps.
        """
        if self.length is not None:
            gap_samples = 0
        else:
            gap_samples = int(self.gap_duration * self.sample_rate)
            
        gap = np.zeros(gap_samples)
        
        parts = []
        for i in range(self.num_repeats):
            parts.append(self.chirp)
            if i < self.num_repeats - 1 and gap_samples > 0:
                parts.append(gap)
                
        return np.concatenate(parts)

    def matched_filter(self, rx_signal: np.ndarray) -> np.ndarray:
        """
        Apply matched filtering via cross-correlation with the template.
        """
        template = self.preamble
        if len(rx_signal) < len(template):
            return np.array([])
        corr = signal.correlate(rx_signal, template, mode='valid')
        return np.abs(corr)

    def normalized_cross_correlation(self, rx_signal: np.ndarray) -> np.ndarray:
        """
        Compute energy-normalized cross-correlation (NCC) bounded in [0, 1].
        Eliminates false triggers caused by microphone noise, sudden clicks,
        or division-by-zero in leading/trailing silence regions.
        """
        template = self.preamble
        if len(rx_signal) < len(template):
            return np.array([])
        corr = np.abs(signal.correlate(rx_signal, template, mode='valid'))
        L = len(template)
        t_energy = np.sqrt(np.sum(template**2))
        win_energy = np.sqrt(np.convolve(rx_signal**2, np.ones(L), mode='valid'))
        safe_energy = np.maximum(win_energy, 0.05 * t_energy)
        ncc = corr / (safe_energy * t_energy)
        return ncc

    def detect(self, rx_signal: np.ndarray, threshold: float = None) -> int | None:
        """
        Detect preamble in received signal via matched filter and peak detection.
        """
        if len(rx_signal) < len(self.preamble):
            return None

        corr = self.matched_filter(rx_signal)
        if len(corr) == 0 or np.max(corr) == 0:
            return None

        if threshold is not None:
            if np.max(corr) < threshold:
                return None
            return int(np.argmax(corr))

        # Compute NCC for robust detection
        ncc = self.normalized_cross_correlation(rx_signal)
        target_th = getattr(self.config, 'correlation_threshold', 0.40)
        effective_th = min(target_th, 0.30)

        if np.max(ncc) >= effective_th:
            return int(np.argmax(ncc))
        return None
