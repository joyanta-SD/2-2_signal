from __future__ import annotations
import numpy as np
from config import AirBuddsConfig, DEFAULT_CONFIG

class AudioWatermark:
    """Spread-Spectrum Audio Watermark embedding and detection."""
    def __init__(self, config=None, sample_rate: int = 44100):
        self.config = config or DEFAULT_CONFIG
        self.sample_rate = sample_rate
        self.seed = 42
        self.amplitude = 0.01
        self.pn_length = 1024
        self.pn_seq = self._generate_pn_sequence()

    def _generate_pn_sequence(self) -> np.ndarray:
        """Generate pseudo-noise sequence."""
        rng = np.random.default_rng(self.seed)
        seq = rng.choice([-1, 1], size=self.pn_length)
        return seq

    def embed(self, signal_in: np.ndarray, watermark_bits: np.ndarray = None) -> np.ndarray:
        """Embed spread-spectrum watermark at low amplitude."""
        y = signal_in.copy()
        # Embed PN sequence periodically
        for i in range(0, len(signal_in) - self.pn_length, self.pn_length):
            y[i:i+self.pn_length] += self.amplitude * self.pn_seq
        return y

    def detect(self, signal_in: np.ndarray) -> tuple[bool, float]:
        """Detect watermark via correlation with PN sequence."""
        max_corr = 0.0
        for i in range(0, len(signal_in) - self.pn_length, self.pn_length // 2):
            segment = signal_in[i:i+self.pn_length]
            corr = np.correlate(segment, self.pn_seq, mode='valid')[0]
            max_corr = max(max_corr, corr)
            
        threshold = self.amplitude * self.pn_length * 0.5
        detected = max_corr > threshold
        return detected, float(max_corr)

    def extract(self, signal_in: np.ndarray) -> np.ndarray | None:
        """Extract watermark bits if detected."""
        detected, _ = self.detect(signal_in)
        if detected:
            # Simple sync indicator extraction
            return np.array([1])
        return None

    def get_pn_sequence(self) -> np.ndarray:
        """Return the PN sequence for analysis."""
        return self.pn_seq
