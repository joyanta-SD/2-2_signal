"""
AirBudds - Audio Hardware Interface
===================================
Hardware abstraction layer for sound card playback and recording using sounddevice.
Includes automatic detection of physical microphones and speakers (bypassing virtual drivers),
hardware volume normalization, and input level metering.
"""

from __future__ import annotations
import numpy as np
import logging
from config import AirBuddsConfig, AudioConfig, DEFAULT_CONFIG

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

class AudioInterface:
    """
    Interface for audio playback and recording.
    Wraps sounddevice with robust device detection, volume scaling, and lead-in padding.
    """
    def __init__(self, config=None):
        """
        Initialize the Audio Interface.
        """
        raw_cfg = config if config is not None else DEFAULT_CONFIG
        self.config = getattr(raw_cfg, 'audio', raw_cfg)
        self.sample_rate = getattr(self.config, 'sample_rate', 44100)
        
        self.device_in = getattr(self.config, 'device_in', None)
        self.device_out = getattr(self.config, 'device_out', None)
        
        if sd is not None:
            self._auto_detect_devices()
            sd.default.samplerate = self.sample_rate
            sd.default.channels = 1
            if self.device_in is not None or self.device_out is not None:
                sd.default.device = (self.device_in, self.device_out)

    def _auto_detect_devices(self) -> None:
        """
        Auto-detect the best physical microphone and speaker, avoiding virtual/inactive devices.
        """
        if sd is None:
            return
            
        try:
            devices = sd.query_devices()
            
            # Find best physical input device if none is explicitly specified
            if self.device_in is None:
                for idx, dev in enumerate(devices):
                    if dev['max_input_channels'] > 0:
                        name = dev['name'].lower()
                        # Skip virtual drivers
                        if any(v in name for v in ['camo', 'mapper', 'virtual', 'stereo mix']):
                            continue
                        # Prefer hardware mics
                        if any(h in name for h in ['microphone array', 'realtek', 'amd', 'intel', 'array']):
                            self.device_in = idx
                            break
                        if self.device_in is None:
                            self.device_in = idx
                            
            # Find best physical output device if none is explicitly specified
            if self.device_out is None:
                for idx, dev in enumerate(devices):
                    if dev['max_output_channels'] > 0:
                        name = dev['name'].lower()
                        if any(v in name for v in ['mapper', 'virtual']):
                            continue
                        if any(h in name for h in ['speaker', 'realtek', 'headphones']):
                            self.device_out = idx
                            break
                        if self.device_out is None:
                            self.device_out = idx
        except Exception as e:
            logging.warning(f"Audio device auto-detection warning: {e}")

    def play(self, signal: np.ndarray, blocking: bool = True, device: int = None) -> None:
        """
        Play audio through the speaker using sounddevice.play().
        """
        if sd is None:
            logging.warning("Cannot play audio: sounddevice is missing.")
            return
            
        max_v = np.max(np.abs(signal))
        # Keep volume at 0.85 to prevent speaker/mic overdrive
        volume = min(getattr(self.config, 'volume', 0.85), 0.85)
        if max_v > 0:
            norm_signal = (signal / max_v) * volume
        else:
            norm_signal = signal
            
        lead_in_sec = getattr(self.config, 'lead_in_seconds', 0.25)
        lead_out_sec = getattr(self.config, 'lead_out_seconds', 0.20)
        lead_in_samples = int(lead_in_sec * self.sample_rate)
        lead_out_samples = int(lead_out_sec * self.sample_rate)
        
        padded_signal = np.pad(norm_signal, (lead_in_samples, lead_out_samples), 'constant').astype(np.float32)
        dev = device if device is not None else self.device_out
        sd.play(padded_signal, samplerate=self.sample_rate, device=dev)
        if blocking:
            sd.wait()
            
    def record(self, duration: float, device: int = None) -> np.ndarray:
        """
        Record from the microphone for the specified duration.
        """
        if sd is None:
            logging.warning("Cannot record audio: sounddevice is missing.")
            return np.zeros(int(duration * self.sample_rate))
            
        frames = int(duration * self.sample_rate)
        dev = device if device is not None else self.device_in
        recording = sd.rec(frames, samplerate=self.sample_rate, channels=1, device=dev, blocking=True)
        return recording.flatten()

    def test_mic_level(self, duration: float = 0.5, device: int = None) -> tuple[float, float]:
        """
        Test microphone input level by recording a short segment.
        Returns:
            tuple[float, float]: (peak_amplitude, rms_amplitude)
        """
        rec = self.record(duration, device=device)
        peak = float(np.max(np.abs(rec)))
        rms = float(np.sqrt(np.mean(rec**2)))
        return peak, rms

    def get_input_devices(self) -> list[tuple[int, str]]:
        """List all available microphone input devices."""
        if sd is None:
            return []
        inputs = []
        for idx, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                api = sd.query_hostapis(dev['hostapi'])['name']
                inputs.append((idx, f"[{idx}] {dev['name']} ({api})"))
        return inputs

    def get_output_devices(self) -> list[tuple[int, str]]:
        """List all available speaker output devices."""
        if sd is None:
            return []
        outputs = []
        for idx, dev in enumerate(sd.query_devices()):
            if dev['max_output_channels'] > 0:
                api = sd.query_hostapis(dev['hostapi'])['name']
                outputs.append((idx, f"[{idx}] {dev['name']} ({api})"))
        return outputs

    def list_devices(self) -> list[dict]:
        if sd is None:
            return []
        return list(sd.query_devices())
        
    def set_device(self, input_device: int = None, output_device: int = None) -> None:
        if input_device is not None:
            self.device_in = input_device
        if output_device is not None:
            self.device_out = output_device
        if sd is not None:
            sd.default.device = (self.device_in, self.device_out)
            
    def get_sample_rate(self) -> int:
        return self.sample_rate
        
    def stop(self) -> None:
        if sd is not None:
            sd.stop()
