from __future__ import annotations
import numpy as np
import logging
from config import AirBuddsConfig, AudioConfig, DEFAULT_CONFIG
import threading
import queue

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

class AudioInterface:
    def __init__(self, config=None):
        raw_cfg = config if config is not None else DEFAULT_CONFIG
        self.config = getattr(raw_cfg, 'audio', raw_cfg)
        self.sample_rate = getattr(self.config, 'sample_rate', 44100)
        self.device_in = getattr(self.config, 'device_in', None)
        self.device_out = getattr(self.config, 'device_out', None)
        
        self._stream = None
        self._audio_queue = queue.Queue()
        
        if sd is not None:
            sd.default.samplerate = self.sample_rate
            sd.default.channels = 1

    def get_input_devices(self) -> list[tuple[int | None, str]]:
        if sd is None: return [(None, "Default System Microphone (Auto)")]
        res = [(None, "Default System Microphone (Auto)")]
        try:
            devs = sd.query_devices()
            res.extend([(idx, f"{dev['name']} [ID:{idx}]") for idx, dev in enumerate(devs) if dev['max_input_channels'] > 0])
        except Exception:
            pass
        return res

    def get_output_devices(self) -> list[tuple[int | None, str]]:
        if sd is None: return [(None, "Default System Speaker (Auto)")]
        res = [(None, "Default System Speaker (Auto)")]
        try:
            devs = sd.query_devices()
            res.extend([(idx, f"{dev['name']} [ID:{idx}]") for idx, dev in enumerate(devs) if dev['max_output_channels'] > 0])
        except Exception:
            pass
        return res


    def play(self, signal: np.ndarray, blocking: bool = True, device: int = None) -> None:
        if sd is None:
            logging.warning("Cannot play audio: sounddevice is missing.")
            return
            
        max_v = np.max(np.abs(signal))
        volume = getattr(self.config, 'volume', 0.95)
        # We cap at 1.0 to prevent hard digital clipping in the output buffer
        volume = min(volume, 1.0)
        
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

    def stop_play(self) -> None:
        """Immediately stop any active playback."""
        if sd is not None:
            sd.stop()

    def start_stream(self, device: int = None):
        """Start infinite background recording."""
        if sd is None:
            raise RuntimeError("Audio hardware unavailable.")
            
        def callback(indata, frames, time, status):
            if status:
                pass
            self._audio_queue.put(indata.copy())
            
        dev = device if device is not None else self.device_in
        self._stream = sd.InputStream(samplerate=self.sample_rate, channels=1, device=dev, callback=callback)
        self._stream.start()
        
    def stop_stream(self) -> np.ndarray:
        """Stop infinite recording and return the full collected array."""
        if self._stream is None:
            return np.array([])
        self._stream.stop()
        self._stream.close()
        self._stream = None
        
        chunks = []
        while not self._audio_queue.empty():
            chunks.append(self._audio_queue.get())
            
        if not chunks:
            return np.array([])
            
        return np.concatenate(chunks).flatten()

    def record(self, duration: float, blocking: bool = True, device: int = None) -> np.ndarray:
        if sd is None:
            logging.warning("Cannot record audio: sounddevice is missing.")
            return np.array([])
            
        frames = int(duration * self.sample_rate)
        dev = device if device is not None else self.device_in
        rec = sd.rec(frames, samplerate=self.sample_rate, channels=1, device=dev, blocking=blocking)
        if blocking:
            return rec.flatten()
        return np.array([])

    def test_mic_level(self, duration: float = 0.5, device: int = None) -> tuple[float, float]:
        if sd is None: return 0.0, 0.0
        frames = int(duration * self.sample_rate)
        dev = device if device is not None else self.device_in
        rec = sd.rec(frames, samplerate=self.sample_rate, channels=1, device=dev, blocking=True)
        rec = rec.flatten()
        peak = float(np.max(np.abs(rec)))
        rms = float(np.sqrt(np.mean(rec**2)))
        return peak, rms
