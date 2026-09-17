from __future__ import annotations
import numpy as np
from config import AirBuddsConfig, DEFAULT_CONFIG
from .preamble import PreambleGenerator

class Synchronizer:

    def __init__(self, config: AirBuddsConfig = None):

        self.config = config if config else DEFAULT_CONFIG
        self.preamble_gen = PreambleGenerator(self.config.preamble, getattr(self.config.audio, 'sample_rate', 44100))
        self.sample_rate = getattr(self.config.audio, 'sample_rate', 44100)

        ofdm = self.config.ofdm
        self.n_fft = getattr(ofdm, 'fft_size', getattr(ofdm, 'n_fft', 1024))
        self.cp_length = getattr(ofdm, 'cp_length', 256)

    def coarse_sync(self, rx_signal: np.ndarray) -> int | None:

        idx = self.preamble_gen.detect(rx_signal)
        if idx is not None:

            return idx + len(self.preamble_gen.preamble)
        return None

    def fine_sync(self, rx_signal: np.ndarray, coarse_offset: int) -> int:

        return coarse_offset

    def estimate_cfo(self, rx_signal: np.ndarray, coarse_idx: int) -> float:

        preamble_len = len(self.preamble_gen.preamble)
        chirp_len = len(self.preamble_gen.chirp)

        preamble_start = coarse_idx - preamble_len
        if preamble_start < 0 or self.preamble_gen.num_repeats < 2:
            return 0.0

        chirp1 = rx_signal[preamble_start : preamble_start + chirp_len]
        chirp2 = rx_signal[preamble_start + chirp_len : preamble_start + 2 * chirp_len]

        if len(chirp1) != len(chirp2) or len(chirp1) == 0:
            return 0.0

        correlation = np.vdot(chirp1, chirp2)
        phase_diff = np.angle(correlation)

        cfo = phase_diff * self.sample_rate / (2 * np.pi * chirp_len)
        return cfo

    def correct_cfo(self, signal: np.ndarray, cfo: float) -> np.ndarray:

        n = np.arange(len(signal))
        correction_factor = np.exp(-1j * 2 * np.pi * cfo * n / self.sample_rate)
        return signal * correction_factor

    def synchronize(self, rx_signal: np.ndarray) -> tuple[np.ndarray, int, float]:

        coarse_idx = self.coarse_sync(rx_signal)
        if coarse_idx is None:
            raise ValueError("Preamble not detected.")

        fine_idx = self.fine_sync(rx_signal, coarse_idx)
        cfo = self.estimate_cfo(rx_signal, fine_idx)
        corrected_signal = self.correct_cfo(rx_signal, cfo)

        return corrected_signal, fine_idx, cfo
