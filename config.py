from dataclasses import dataclass, field
import numpy as np
from typing import List

@dataclass
class AudioConfig:
    sample_rate: int = 44_100
    device_in: int | None = None
    device_out: int | None = None
    volume: float = 0.95
    lead_in_seconds: float = 0.25
    lead_out_seconds: float = 0.20

@dataclass
class OFDMConfig:
    fft_size: int = 1024
    cp_length: int = 512
    num_data_subcarriers: int = 32
    num_pilot_subcarriers: int = 8
    pilot_value: complex = 1.0 + 0j
    guard_band_left: int = 25
    guard_band_right: int = 420
    dc_null: bool = True
    symbol_duration: float = field(init=False)
    subcarrier_spacing: float = field(init=False)

    def __post_init__(self) -> None:
        self.subcarrier_spacing = 44_100 / self.fft_size
        self.symbol_duration = (self.fft_size + self.cp_length) / 44_100

    def pilot_indices(self) -> np.ndarray:
        start = self.guard_band_left + 1
        stop = (self.fft_size // 2) - self.guard_band_right - 1
        return np.linspace(start, stop, self.num_pilot_subcarriers, dtype=int)

    def data_indices(self) -> np.ndarray:
        pilots = set(self.pilot_indices())
        start = self.guard_band_left + 1
        stop = (self.fft_size // 2) - self.guard_band_right
        all_usable = [k for k in range(start, stop) if k not in pilots and k != 0]
        return np.array(all_usable[:self.num_data_subcarriers], dtype=int)

@dataclass
class QAMConfig:
    order: int = 4
    normalize: bool = True
    use_gray_code: bool = True
    M: int | None = None
    def __post_init__(self) -> None:
        if self.M is not None:
            self.order = self.M
    @property
    def bits_per_symbol(self) -> int:
        return int(np.log2(self.order))

@dataclass
class PreambleConfig:
    f_start: float = 800.0
    f_stop: float = 3_200.0
    duration: float = 0.30
    num_repeats: int = 2
    correlation_threshold: float = 0.5
    window: str = "hann"
    type: str = "chirp"
    length: int | None = None
    def __post_init__(self) -> None:
        if self.length is not None and self.length > 0:
            self.duration = self.length / 44100.0

@dataclass
class CRCConfig:
    polynomial: int = 0x04C11DB7
    init_value: int = 0xFFFFFFFF
    final_xor: int = 0xFFFFFFFF
    width: int = 32

@dataclass
class ProtocolConfig:
    max_retries: int = 5
    ack_timeout: float = 2.0
    header_bits: int = 32
    max_payload_bytes: int = 256
    use_compression: bool = False

@dataclass
class AirBuddsConfig:
    version: str = "4.0-Hybrid"
    team_name: str = "8J3D9"
    members: tuple = ("Joyanta Sutradhar (2305083)", "Dipbroto Karmokar Dip (2305089)")
    
    audio: AudioConfig = field(default_factory=AudioConfig)
    ofdm: OFDMConfig = field(default_factory=OFDMConfig)
    qam: QAMConfig = field(default_factory=QAMConfig)
    preamble: PreambleConfig = field(default_factory=PreambleConfig)
    crc: CRCConfig = field(default_factory=CRCConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    
    morse_freq: float = 800.0
    morse_wpm: float = 20.0
    mode: str = "morse"

DEFAULT_CONFIG = AirBuddsConfig()
