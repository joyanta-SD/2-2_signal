from __future__ import annotations
import numpy as np
import scipy.io.wavfile as wav
import logging
from config import AirBuddsConfig, DEFAULT_CONFIG

try:
    import sounddevice as sd
except ImportError:
    sd = None
    logging.warning("sounddevice module not found. Playback disabled.")

class Player:
    """
    Playback Buffer class for audio transmission.
    """
    def __init__(self, config: AirBuddsConfig = None):
        """
        Initialize the Player.
        
        Parameters:
            config: AirBudds configuration.
        """
        raw_cfg = config if config is not None else DEFAULT_CONFIG
        self.config = getattr(raw_cfg, 'audio', raw_cfg)
        self.sample_rate = getattr(self.config, 'sample_rate', 44100)
        self.volume = 1.0
        self.is_active = False
        
    def play(self, signal: np.ndarray, blocking: bool = True) -> None:
        """
        Play an audio signal.
        
        Parameters:
            signal: Signal array to play.
            blocking: If True, blocks until playback completes.
        """
        if sd is None:
            return
            
        scaled_signal = np.clip(signal * self.volume, -1.0, 1.0).astype(np.float32)
        self.is_active = True
        try:
            sd.play(scaled_signal, samplerate=self.sample_rate)
            if blocking:
                sd.wait()
                self.is_active = False
        except Exception as e:
            logging.error(f"Error during playback: {e}")
            self.is_active = False
            
    def play_async(self, signal: np.ndarray) -> None:
        """
        Non-blocking playback of an audio signal.
        
        Parameters:
            signal: Signal array to play.
        """
        self.play(signal, blocking=False)
        
    def stop(self) -> None:
        """Stop playback immediately."""
        if sd is not None:
            sd.stop()
        self.is_active = False
        
    def is_playing(self) -> bool:
        """
        Check if audio is currently playing.
        
        Returns:
            bool: True if playback is active.
        """
        # sounddevice lacks a direct is_playing() for simple play(), 
        # but we can rely on our internal state or wait() state.
        return self.is_active
        
    def set_volume(self, volume: float) -> None:
        """
        Set playback volume scale.
        
        Parameters:
            volume: Volume level between 0.0 and 1.0.
        """
        self.volume = max(0.0, min(1.0, volume))
        
    def save_wav(self, signal: np.ndarray, filepath: str) -> None:
        """
        Export an audio signal to a WAV file.
        
        Parameters:
            signal: Audio signal to save.
            filepath: Path to output WAV file.
        """
        scaled_signal = np.clip(signal * self.volume, -1.0, 1.0).astype(np.float32)
        wav.write(filepath, self.sample_rate, scaled_signal)
