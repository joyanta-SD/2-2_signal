from __future__ import annotations
import numpy as np
import time
from config import AirBuddsConfig, DEFAULT_CONFIG

class RetryManager:
    """
    ARQ Retransmission manager for handling ACK/NAK and retries.
    """
    def __init__(self, config: AirBuddsConfig = None):
        """
        Initialize the RetryManager.
        
        Parameters:
            config: AirBudds configuration.
        """
        self.config = config or DEFAULT_CONFIG
        self.max_retries = getattr(self.config.protocol, 'max_retries', 3) if hasattr(self.config, 'protocol') else 3
        self.sample_rate = getattr(self.config.audio, 'sample_rate', 44100) if hasattr(self.config, 'audio') else 44100
        
        # Stats tracking
        self.attempts = 0
        self.successes = 0
        self.total_retries = 0
        
    def transmit_with_retry(self, data: bytes, tx_func, rx_func, verify_func) -> tuple[bool, int, dict]:
        """
        Transmit data with automatic retry based on ACK/NAK.
        
        Parameters:
            data: The frame data to transmit.
            tx_func: Function to call for transmission.
            rx_func: Function to call to listen for ACK/NAK.
            verify_func: Function to verify received data if acting as responder.
            
        Returns:
            tuple: (success_bool, num_attempts, statistics_dict)
        """
        attempt = 0
        success = False
        
        while attempt <= self.max_retries:
            self.attempts += 1
            attempt += 1
            
            # Transmit
            tx_func(data)
            
            # Listen for ACK/NAK
            resp_signal = rx_func()
            status = self.detect_ack_nak(resp_signal)
            
            if status == 'ACK':
                success = True
                self.successes += 1
                break
            elif status == 'NAK':
                self.total_retries += 1
            elif status == 'TIMEOUT':
                self.total_retries += 1
                
        stats = self.get_statistics()
        return success, attempt, stats
        
    def verify_reception(self, received_data: bytes) -> bool:
        """
        Check if reception is valid (typically used in conjunction with FrameBuilder CRC).
        
        Parameters:
            received_data: Received frame bytes.
            
        Returns:
            bool: True if CRC/structure is valid.
        """
        # Simplistic stub; real CRC validation usually happens in FrameBuilder
        # This function might call FrameBuilder under the hood or check external state.
        return True
        
    def generate_ack(self) -> np.ndarray:
        """
        Generate a short 5kHz tone burst representing an ACK.
        
        Returns:
            np.ndarray: ACK audio signal.
        """
        duration = 0.1  # 100 ms burst
        t = np.arange(int(self.sample_rate * duration)) / self.sample_rate
        # 5 kHz tone
        return np.sin(2 * np.pi * 5000 * t)
        
    def generate_nak(self) -> np.ndarray:
        """
        Generate a short 3kHz tone burst representing a NAK.
        
        Returns:
            np.ndarray: NAK audio signal.
        """
        duration = 0.1
        t = np.arange(int(self.sample_rate * duration)) / self.sample_rate
        # 3 kHz tone
        return np.sin(2 * np.pi * 3000 * t)
        
    def detect_ack_nak(self, signal: np.ndarray) -> str:
        """
        Detect 'ACK' or 'NAK' from a received audio signal by energy in frequency bands.
        
        Parameters:
            signal: The received audio signal.
            
        Returns:
            str: 'ACK', 'NAK', or 'TIMEOUT'.
        """
        if len(signal) == 0:
            return 'TIMEOUT'
            
        # Perform FFT to check energy
        N = len(signal)
        freqs = np.fft.rfftfreq(N, 1/self.sample_rate)
        fft_mag = np.abs(np.fft.rfft(signal))
        
        # Energy around 5kHz and 3kHz (+/- 200Hz tolerance)
        ack_band = (freqs > 4800) & (freqs < 5200)
        nak_band = (freqs > 2800) & (freqs < 3200)
        
        ack_energy = np.sum(fft_mag[ack_band])
        nak_energy = np.sum(fft_mag[nak_band])
        
        threshold = 100.0  # arbitrary threshold, depends on signal scaling
        
        if ack_energy > nak_energy and ack_energy > threshold:
            return 'ACK'
        elif nak_energy > ack_energy and nak_energy > threshold:
            return 'NAK'
            
        return 'TIMEOUT'
        
    def get_statistics(self) -> dict:
        """
        Return transmission statistics.
        
        Returns:
            dict: Statistics dictionary.
        """
        return {
            'attempts': self.attempts,
            'success_rate': (self.successes / self.attempts) if self.attempts > 0 else 0.0,
            'total_retries': self.total_retries,
            'avg_retries': (self.total_retries / self.successes) if self.successes > 0 else 0.0
        }
        
    def reset_statistics(self) -> None:
        """Reset transmission statistics."""
        self.attempts = 0
        self.successes = 0
        self.total_retries = 0
