from __future__ import annotations
import numpy as np
from scipy import signal
from config import AirBuddsConfig, DEFAULT_CONFIG

class RIRSimulator:
    """Room Impulse Response Simulator."""
    def __init__(self, config=None, sample_rate: int = 44100):
        self.config = config or DEFAULT_CONFIG
        self.sample_rate = sample_rate
        self.h = None
        rir_cfg = getattr(self.config, 'rir', self.config)
        self.preset = getattr(rir_cfg, 'preset', 'small_room')
        
    def generate_impulse_response(self) -> np.ndarray:
        """Create synthetic RIR h[n] with multiple reflections."""
        params = self._preset_parameters(self.preset)
        num_reflections = params['num_reflections']
        max_delay_ms = params['max_delay_ms']
        decay = params['decay_factor']
        
        max_samples = int((max_delay_ms / 1000.0) * self.sample_rate)
        self.h = np.zeros(max_samples + 1)
        
        # Direct path
        self.h[0] = 1.0
        
        # Reflections
        if num_reflections > 0:
            delays = np.random.choice(range(1, max_samples), num_reflections, replace=False)
            for d in delays:
                # h[n] = sum( a_k * delta[n - tau_k] )
                a = (decay ** (d / max_samples)) * np.random.normal(0, 0.1)
                self.h[d] = a
                
        return self.h

    def _preset_parameters(self, preset: str) -> dict:
        """Return preset configuration for reflections."""
        presets = {
            'anechoic': {'num_reflections': 0, 'max_delay_ms': 1, 'decay_factor': 0.0},
            'small_room': {'num_reflections': 5, 'max_delay_ms': 20, 'decay_factor': 0.5},
            'hallway': {'num_reflections': 9, 'max_delay_ms': 50, 'decay_factor': 0.6},
            'open_air': {'num_reflections': 2, 'max_delay_ms': 10, 'decay_factor': 0.3}
        }
        return presets.get(preset, presets['small_room'])

    def apply_channel(self, signal_in: np.ndarray) -> np.ndarray:
        """Apply channel impulse response via convolution."""
        if self.h is None:
            self.generate_impulse_response()
        return signal.fftconvolve(signal_in, self.h, mode='full')[:len(signal_in)]

    def add_noise(self, signal_in: np.ndarray, snr_db: float = None) -> np.ndarray:
        """Add AWGN at specified SNR."""
        if snr_db is None:
            snr_db = 20.0
        sig_power = np.mean(signal_in**2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = sig_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal_in))
        return signal_in + noise

    def simulate(self, signal_in: np.ndarray) -> np.ndarray:
        """Full simulation: apply channel + add noise."""
        y = self.apply_channel(signal_in)
        return self.add_noise(y)

    def get_impulse_response(self) -> np.ndarray:
        """Return the generated impulse response."""
        if self.h is None:
            self.generate_impulse_response()
        return self.h

    def plot_impulse_response(self):
        """Plot impulse response stem plot."""
        try:
            import matplotlib.pyplot as plt
            h = self.get_impulse_response()
            plt.figure()
            plt.stem(h)
            plt.title("Room Impulse Response")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.show()
        except ImportError:
            pass
