from __future__ import annotations
import numpy as np
from config import AirBuddsConfig, DEFAULT_CONFIG
from .preamble import PreambleGenerator

class Synchronizer:
    """
    Timing and Frequency Synchronizer for OFDM.
    Handles coarse sync (via preamble), fine sync (via Cyclic Prefix), and Carrier Frequency Offset (CFO) estimation/correction.
    """
    def __init__(self, config: AirBuddsConfig = None):
        """
        Initialize Synchronizer.
        
        Parameters:
            config (AirBuddsConfig): Main config.
        """
        self.config = config if config else DEFAULT_CONFIG
        self.preamble_gen = PreambleGenerator(self.config.preamble, getattr(self.config.audio, 'sample_rate', 44100))
        self.sample_rate = getattr(self.config.audio, 'sample_rate', 44100)
        
        # OFDM parameters — OFDMConfig uses 'fft_size' / 'cp_length'
        ofdm = self.config.ofdm
        self.n_fft = getattr(ofdm, 'fft_size', getattr(ofdm, 'n_fft', 1024))
        self.cp_length = getattr(ofdm, 'cp_length', 256)

    def coarse_sync(self, rx_signal: np.ndarray) -> int | None:
        """
        Coarse timing synchronization using preamble matched filter.
        
        Parameters:
            rx_signal (np.ndarray): Received time-domain signal.
            
        Returns:
            int: Start index of the OFDM data frame (after preamble).
        """
        idx = self.preamble_gen.detect(rx_signal)
        if idx is not None:
            # The matched filter peak gives the start of the preamble.
            # We want to return the start of the actual data, which is after the preamble length.
            return idx + len(self.preamble_gen.preamble)
        return None

    def fine_sync(self, rx_signal: np.ndarray, coarse_offset: int) -> int:
        """
        Fine timing synchronization.
        The LFM chirp matched filter already achieves sample-accurate synchronization.
        """
        return coarse_offset

    def estimate_cfo(self, rx_signal: np.ndarray, start_idx: int) -> float:
        """
        Estimate Carrier Frequency Offset from the cyclic prefix of the first OFDM symbol.
        Phase difference between CP and end of symbol = 2π * CFO * N_FFT / Fs
        
        Parameters:
            rx_signal (np.ndarray): Received time-domain signal.
            start_idx (int): Fine sync index.
            
        Returns:
            float: Estimated Carrier Frequency Offset (Hz).
        """
        if start_idx + self.n_fft + self.cp_length > len(rx_signal):
            return 0.0
            
        cp_part = rx_signal[start_idx : start_idx + self.cp_length]
        end_part = rx_signal[start_idx + self.n_fft : start_idx + self.n_fft + self.cp_length]
        
        # Phase difference: arg(sum(cp_part * conj(end_part)))
        correlation = np.vdot(end_part, cp_part)
        phase_diff = np.angle(correlation)
        
        # phase_diff = 2 * pi * CFO * T_symbol
        # T_symbol = N_FFT / Fs (time between the two correlated parts)
        cfo = phase_diff * self.sample_rate / (2 * np.pi * self.n_fft)
        return cfo

    def correct_cfo(self, signal: np.ndarray, cfo: float) -> np.ndarray:
        """
        Correct Carrier Frequency Offset in time-domain signal.
        Multiply by exp(-j2πΔf·n/Fs)
        
        Parameters:
            signal (np.ndarray): Time-domain signal.
            cfo (float): CFO to correct (Hz).
            
        Returns:
            np.ndarray: Corrected signal.
        """
        n = np.arange(len(signal))
        correction_factor = np.exp(-1j * 2 * np.pi * cfo * n / self.sample_rate)
        return signal * correction_factor

    def synchronize(self, rx_signal: np.ndarray) -> tuple[np.ndarray, int, float]:
        """
        Full synchronization pipeline.
        
        Parameters:
            rx_signal (np.ndarray): Received raw time-domain signal.
            
        Returns:
            tuple: (CFO corrected signal, start index, cfo_estimate)
        """
        coarse_idx = self.coarse_sync(rx_signal)
        if coarse_idx is None:
            raise ValueError("Preamble not detected.")
            
        fine_idx = self.fine_sync(rx_signal, coarse_idx)
        cfo = self.estimate_cfo(rx_signal, fine_idx)
        corrected_signal = self.correct_cfo(rx_signal, cfo)
        
        return corrected_signal, fine_idx, cfo
