from __future__ import annotations
import numpy as np
import scipy.io.wavfile as wav
import logging
from config import AirBuddsConfig, DEFAULT_CONFIG

try:
    import sounddevice as sd
except ImportError:
    sd = None
    logging.warning("sounddevice module not found. Recording disabled.")

class Recorder:
    """
    Recording Buffer class for continuous and fixed-duration recording.
    """
    def __init__(self, config: AirBuddsConfig = None):
        """
        Initialize the Recorder.
        
        Parameters:
            config: AirBudds configuration.
        """
        raw_cfg = config if config is not None else DEFAULT_CONFIG
        self.config = getattr(raw_cfg, 'audio', raw_cfg)
        self.sample_rate = getattr(self.config, 'sample_rate', 44100)
        self.buffer = []
        self.stream = None
        self.is_recording = False
        
    def _audio_callback(self, indata: np.ndarray, frames: int, time, status):
        """Callback to receive audio data from the input stream."""
        if status:
            logging.warning(f"Audio callback status: {status}")
        self.buffer.append(indata.copy())
        
    def start_recording(self) -> None:
        """Start continuous recording using a callback-based stream."""
        if sd is None:
            return
            
        self.buffer = []
        self.is_recording = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback
        )
        self.stream.start()
        
    def stop_recording(self) -> np.ndarray:
        """
        Stop continuous recording and return the recorded buffer.
        
        Returns:
            np.ndarray: The complete recorded signal.
        """
        if not self.is_recording or self.stream is None:
            return np.array([])
            
        self.stream.stop()
        self.stream.close()
        self.stream = None
        self.is_recording = False
        
        if not self.buffer:
            return np.array([])
            
        full_recording = np.concatenate(self.buffer).flatten()
        return full_recording
        
    def record_duration(self, duration: float) -> np.ndarray:
        """
        Record for a fixed duration.
        
        Parameters:
            duration: Duration in seconds.
            
        Returns:
            np.ndarray: Recorded audio signal.
        """
        if sd is None:
            return np.zeros(int(duration * self.sample_rate))
            
        frames = int(duration * self.sample_rate)
        recording = sd.rec(frames, samplerate=self.sample_rate, channels=1, blocking=True)
        return recording.flatten()
        
    def save_wav(self, signal: np.ndarray, filepath: str) -> None:
        """
        Save an audio signal to a WAV file.
        
        Parameters:
            signal: Audio signal to save.
            filepath: Path to save the file.
        """
        # Scipy wavfile expects int16 or float32. We'll use float32 [-1, 1].
        norm_signal = np.clip(signal, -1.0, 1.0).astype(np.float32)
        wav.write(filepath, self.sample_rate, norm_signal)
        
    def load_wav(self, filepath: str) -> np.ndarray:
        """
        Load a WAV file and normalize to float32.
        
        Parameters:
            filepath: Path to the WAV file.
            
        Returns:
            np.ndarray: Loaded audio signal normalized to [-1, 1].
        """
        rate, data = wav.read(filepath)
        if rate != self.sample_rate:
            logging.warning(f"Loaded WAV sample rate {rate} differs from config {self.sample_rate}")
            
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
            
        if data.ndim > 1:
            data = data[:, 0]  # Take only first channel if stereo
            
        return data.flatten()
        
    def get_buffer(self) -> np.ndarray:
        """
        Return current recording buffer without stopping.
        
        Returns:
            np.ndarray: Current recorded signal.
        """
        if not self.buffer:
            return np.array([])
        return np.concatenate(self.buffer).flatten()
