from __future__ import annotations
import numpy as np
from scipy import signal
from config import AirBuddsConfig, DEFAULT_CONFIG
from dsp.filters import BandpassFilter, LowpassFilter

class SpectralShifter:
    """Ultrasonic Frequency Shifting pipeline."""
    def __init__(self, config=None, sample_rate: int = 44100):
        self.config = config or DEFAULT_CONFIG
        self.sample_rate = sample_rate
        self.fc = 19000.0 # 19 kHz
        
        self.bp_filter = BandpassFilter(18000, 20000, sample_rate=self.sample_rate, order=101)
        self.lp_filter = LowpassFilter(2000, sample_rate=self.sample_rate, order=101)

    def shift_up(self, baseband_signal: np.ndarray) -> np.ndarray:
        """Multiply by carrier to upshift to ultrasonic band."""
        n = np.arange(len(baseband_signal))
        # x_up(t) = x(t) * 2 * cos(2pi * fc * t)
        carrier = 2 * np.cos(2 * np.pi * self.fc * n / self.sample_rate)
        return baseband_signal * carrier

    def shift_down(self, passband_signal: np.ndarray) -> np.ndarray:
        """Multiply by carrier then low-pass filter to recover baseband."""
        n = np.arange(len(passband_signal))
        carrier = np.cos(2 * np.pi * self.fc * n / self.sample_rate)
        mixed = passband_signal * carrier
        return self.lp_filter.apply(mixed)

    def apply_bandpass(self, signal_in: np.ndarray) -> np.ndarray:
        """Bandpass filter around ultrasonic band."""
        return self.bp_filter.apply(signal_in)

    def encode_ultrasonic(self, baseband_signal: np.ndarray) -> np.ndarray:
        """Full pipeline: shift up -> bandpass."""
        shifted = self.shift_up(baseband_signal)
        return self.apply_bandpass(shifted)

    def decode_ultrasonic(self, ultrasonic_signal: np.ndarray) -> np.ndarray:
        """Full pipeline: bandpass -> shift down -> lowpass."""
        bandpassed = self.apply_bandpass(ultrasonic_signal)
        return self.shift_down(bandpassed)

    def is_ultrasonic(self, signal_in: np.ndarray) -> bool:
        """Detect if signal has energy in ultrasonic band."""
        filtered = self.apply_bandpass(signal_in)
        power_in_band = np.mean(filtered**2)
        power_total = np.mean(signal_in**2) + 1e-12
        return (power_in_band / power_total) > 0.1 # Example threshold
