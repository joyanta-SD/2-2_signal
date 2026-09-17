import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
from core.ofdm_transceiver import OFDMTransceiver
from config import AirBuddsConfig

class SteganographyTransceiver:

    def __init__(self, config: AirBuddsConfig = None):
        import copy
        self.config = copy.deepcopy(config) if config else AirBuddsConfig()

        self.config.ofdm.guard_band_left = 350
        self.config.ofdm.guard_band_right = 10

        if hasattr(self.config, 'preamble'):
            self.config.preamble.use_noise_preamble = True
            self.config.preamble.f_start = 15000.0
            self.config.preamble.f_stop = 20000.0

        self.ofdm = OFDMTransceiver(self.config)

        self.stego_ratio = 0.35  

    def encode(self, data: bytes) -> np.ndarray:

        return self.ofdm.encode(data)

    def encode_with_carrier(self, data: bytes, carrier_filepath: str) -> np.ndarray:

        ofdm_signal = self.ofdm.encode(data)

        try:
            sr, carrier = wavfile.read(carrier_filepath)

            if carrier.ndim > 1:
                carrier = np.mean(carrier, axis=-1)
            carrier = carrier.astype(np.float32)

            max_c = np.max(np.abs(carrier))
            if max_c > 0:
                carrier = carrier / max_c

            if sr != self.config.audio.sample_rate:
                import math
                gcd = math.gcd(sr, self.config.audio.sample_rate)
                carrier = signal.resample_poly(carrier, self.config.audio.sample_rate // gcd, sr // gcd).astype(np.float32)

        except Exception as e:
            raise ValueError(f"Failed to load carrier audio: {e}")

        max_len = max(len(ofdm_signal), len(carrier))
        combined = np.zeros(max_len, dtype=np.float32)

        combined[:len(carrier)] += carrier * 0.95

        combined[:len(ofdm_signal)] += ofdm_signal * self.stego_ratio

        max_comb = np.max(np.abs(combined))
        if max_comb > 1.0:
            combined = combined / max_comb

        return combined

    def decode(self, rx_signal: np.ndarray) -> tuple[bytes, dict]:

        return self.ofdm.decode(rx_signal)

    def get_tx_signal(self) -> np.ndarray | None:
        return self.ofdm.get_tx_signal()

    def get_rx_constellation(self) -> tuple[np.ndarray, np.ndarray] | None:
        return self.ofdm.get_rx_constellation()

    def get_channel_estimate(self) -> tuple[np.ndarray, np.ndarray] | None:
        return self.ofdm.get_channel_estimate()
