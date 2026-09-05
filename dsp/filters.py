from __future__ import annotations
import numpy as np
from scipy import signal
from config import AirBuddsConfig, DEFAULT_CONFIG

class DigitalFilter:
    """Base class for digital filters."""
    def __init__(self, sample_rate: int = 44100, order: int = 101, method: str = 'fir'):
        self.sample_rate = sample_rate
        self.order = order
        self.method = method
        self.b = None
        self.a = None

    def design(self) -> np.ndarray:
        """Design the filter coefficients."""
        raise NotImplementedError

    def apply(self, signal_in: np.ndarray) -> np.ndarray:
        """Apply the digital filter to a signal."""
        if self.b is None:
            self.design()
        if self.method == 'fir':
            return signal.lfilter(self.b, 1.0, signal_in)
        else:
            return signal.filtfilt(self.b, self.a, signal_in)

    def frequency_response(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the frequency response of the filter."""
        if self.b is None:
            self.design()
        w, h = signal.freqz(self.b, self.a if self.a is not None else 1.0, fs=self.sample_rate)
        return w, 20 * np.log10(np.abs(h) + 1e-12)
        
    def plot_response(self):
        """Plot the frequency response of the filter."""
        try:
            import matplotlib.pyplot as plt
            w, mag = self.frequency_response()
            plt.figure()
            plt.plot(w, mag)
            plt.title("Filter Frequency Response")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")
            plt.grid(True)
            plt.show()
        except ImportError:
            pass

class LowpassFilter(DigitalFilter):
    """Lowpass digital filter."""
    def __init__(self, cutoff_freq: float, sample_rate: int = 44100, order: int = 101, method: str = 'fir'):
        super().__init__(sample_rate, order, method)
        self.cutoff_freq = cutoff_freq

    def design(self) -> np.ndarray:
        nyq = 0.5 * self.sample_rate
        norm_cutoff = self.cutoff_freq / nyq
        if self.method == 'fir':
            self.b = signal.firwin(self.order, norm_cutoff, window='hamming')
            self.a = np.array([1.0])
        else:
            self.b, self.a = signal.butter(self.order, norm_cutoff, btype='low')
        return self.b

class HighpassFilter(DigitalFilter):
    """Highpass digital filter."""
    def __init__(self, cutoff_freq: float, sample_rate: int = 44100, order: int = 101, method: str = 'fir'):
        super().__init__(sample_rate, order, method)
        self.cutoff_freq = cutoff_freq

    def design(self) -> np.ndarray:
        nyq = 0.5 * self.sample_rate
        norm_cutoff = self.cutoff_freq / nyq
        if self.method == 'fir':
            self.b = signal.firwin(self.order, norm_cutoff, window='hamming', pass_zero=False)
            self.a = np.array([1.0])
        else:
            self.b, self.a = signal.butter(self.order, norm_cutoff, btype='high')
        return self.b

class BandpassFilter(DigitalFilter):
    """Bandpass digital filter."""
    def __init__(self, low_freq: float, high_freq: float, sample_rate: int = 44100, order: int = 101, method: str = 'fir'):
        super().__init__(sample_rate, order, method)
        self.low_freq = low_freq
        self.high_freq = high_freq

    def design(self) -> np.ndarray:
        nyq = 0.5 * self.sample_rate
        low = self.low_freq / nyq
        high = self.high_freq / nyq
        if self.method == 'fir':
            self.b = signal.firwin(self.order, [low, high], window='hamming', pass_zero=False)
            self.a = np.array([1.0])
        else:
            self.b, self.a = signal.butter(self.order, [low, high], btype='bandpass')
        return self.b
