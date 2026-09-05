"""
AirBudds - Global Configuration and System Parameters
=====================================================
Course  : Signals and Linear Systems
Team    : 8J3D9
Members : Joyanta Sutradhar (2305083), Dipbroto Karmokar Dip (2305089)

Central configuration module that defines every tuneable parameter for the
AirBudds acoustic OFDM transceiver. Tuned specifically for audible, robust
over-the-air sound transmission through standard laptop speakers and microphones.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Any


# ------------------------------------------------------------------------------
#  Audio Hardware
# ------------------------------------------------------------------------------
@dataclass
class AudioConfig:
    """Parameters that govern the sound-card interface."""

    sample_rate: int = 44_100          # Hz - CD-quality audio
    bit_depth: int = 16                # bits per sample (signed PCM)
    channels: int = 1                  # mono
    chunk_size: int = 1024             # frames per buffer
    dtype: str = "float32"             # numpy dtype used internally
    device_in: int | None = None       # None -> system default mic
    device_out: int | None = None      # None -> system default speaker
    playback_blocking: bool = True     # block until playback finishes
    volume: float = 0.95               # loud and clear output amplitude
    lead_in_seconds: float = 0.25      # DAC wake-up lead-in padding (250 ms)
    lead_out_seconds: float = 0.20     # trailing padding


# ------------------------------------------------------------------------------
#  OFDM Physical Layer (Audible Band: 1.0 kHz - 4.0 kHz)
# ------------------------------------------------------------------------------
@dataclass
class OFDMConfig:
    """Orthogonal Frequency Division Multiplexing parameters in the audible band.

    Subcarriers are placed strictly within 1.1 kHz - 3.9 kHz for maximum
    acoustic output on laptop speakers and optimal human audibility.
    """

    fft_size: int = 1024               # N - number of IFFT/FFT points
    cp_length: int = 512               # L_cp - cyclic prefix length (~11.6 ms to eliminate echoes)
    num_data_subcarriers: int = 32     # subcarriers carrying payload QAM symbols
    num_pilot_subcarriers: int = 8     # known-value subcarriers for channel est.
    pilot_value: complex = 1.0 + 0j    # BPSK pilot = +1
    guard_band_left: int = 25          # starts at ~1.1 kHz (clearly audible)
    guard_band_right: int = 420        # ends at ~3.9 kHz (sweet spot of laptop speakers)
    dc_null: bool = True               # zero the DC subcarrier (index 0)

    # Derived helpers
    symbol_duration: float = field(init=False)
    subcarrier_spacing: float = field(init=False)

    def __post_init__(self) -> None:
        self.subcarrier_spacing = 44_100 / self.fft_size          # Delta f approx 43.07 Hz
        self.symbol_duration = (self.fft_size + self.cp_length) / 44_100  # seconds

    @property
    def total_used_subcarriers(self) -> int:
        return self.num_data_subcarriers + self.num_pilot_subcarriers

    def pilot_indices(self) -> np.ndarray:
        """Return sorted subcarrier indices reserved for pilots."""
        start = self.guard_band_left + 1
        stop = (self.fft_size // 2) - self.guard_band_right - 1
        return np.linspace(start, stop, self.num_pilot_subcarriers, dtype=int)

    def data_indices(self) -> np.ndarray:
        """Return subcarrier indices available for data symbols."""
        pilots = set(self.pilot_indices())
        start = self.guard_band_left + 1
        stop = (self.fft_size // 2) - self.guard_band_right
        all_usable = [k for k in range(start, stop) if k not in pilots and k != 0]
        return np.array(all_usable[:self.num_data_subcarriers], dtype=int)


# ------------------------------------------------------------------------------
#  QAM Modulation
# ------------------------------------------------------------------------------
@dataclass
class QAMConfig:
    """Quadrature Amplitude Modulation constellation parameters."""

    order: int = 4                     # M-QAM order: 4 (QPSK), 16, or 64
    normalize: bool = True             # scale constellation to unit avg power
    use_gray_code: bool = True         # Gray-coded bit<->symbol mapping
    M: int | None = None

    def __post_init__(self) -> None:
        if self.M is not None:
            self.order = self.M

    @property
    def bits_per_symbol(self) -> int:
        """Number of bits carried by each constellation point: log2(M)."""
        return int(np.log2(self.order))


# ------------------------------------------------------------------------------
#  Preamble / Synchronisation (Rich Audible Chirp)
# ------------------------------------------------------------------------------
@dataclass
class PreambleConfig:
    """Linear Frequency Modulated (LFM) chirp preamble parameters (Audible Range)."""

    f_start: float = 800.0             # Hz - chirp start frequency (Audible)
    f_stop: float = 3_200.0            # Hz - chirp stop frequency (Audible)
    duration: float = 0.30             # seconds (300 ms - rich audible tone)
    num_repeats: int = 1               # clean single preamble
    correlation_threshold: float = 0.5 # normalised peak threshold for detection
    window: str = "hann"               # amplitude tapering to reduce spectral splash
    type: str = "chirp"
    length: int | None = None

    def __post_init__(self) -> None:
        if self.length is not None and self.length > 0:
            self.duration = self.length / 44100.0


# ------------------------------------------------------------------------------
#  Ultrasonic / Inaudible Mode (Retained for experimental study)
# ------------------------------------------------------------------------------
@dataclass
class UltrasonicConfig:
    """Parameters for ultrasonic experiments (disabled by default)."""

    enabled: bool = False
    carrier_freq: float = 19_000.0
    bandwidth: float = 2_000.0
    filter_order: int = 101
    fft_size: int = 256
    cp_length: int = 64


# ------------------------------------------------------------------------------
#  CRC / Error Detection
# ------------------------------------------------------------------------------
@dataclass
class CRCConfig:
    """CRC-32 polynomial configuration."""

    polynomial: int = 0x04C11DB7
    init_value: int = 0xFFFFFFFF
    final_xor: int = 0xFFFFFFFF
    width: int = 32


# ------------------------------------------------------------------------------
#  LMS Adaptive Filter
# ------------------------------------------------------------------------------
@dataclass
class LMSConfig:
    """Least Mean Squares adaptive noise-cancellation parameters."""

    filter_order: int = 64
    step_size: float = 0.01
    leakage: float = 1.0
    noise_estimation_ms: float = 200.0
    num_taps: int | None = None
    mu: float | None = None

    def __post_init__(self) -> None:
        if self.num_taps is not None:
            self.filter_order = self.num_taps
        if self.mu is not None:
            self.step_size = self.mu


# ------------------------------------------------------------------------------
#  RIR Simulator
# ------------------------------------------------------------------------------
@dataclass
class RIRConfig:
    """Room Impulse Response simulation parameters."""

    preset: str = "small_room"
    num_reflections: int = 6
    max_delay_ms: float = 50.0
    decay_factor: float = 0.6
    snr_db: float = 30.0
    direct_path_gain: float = 1.0


# ------------------------------------------------------------------------------
#  Watermark
# ------------------------------------------------------------------------------
@dataclass
class WatermarkConfig:
    """Spread-spectrum audio watermark settings."""

    enabled: bool = False
    pn_length: int = 1023
    pn_seed: int = 42
    amplitude: float = 0.005
    detection_threshold: float = 0.5


# ------------------------------------------------------------------------------
#  Protocol / Link Layer
# ------------------------------------------------------------------------------
@dataclass
class ProtocolConfig:
    """Automatic Repeat reQuest (ARQ) and framing parameters."""

    max_retries: int = 5
    ack_timeout: float = 2.0
    header_bits: int = 32
    max_payload_bytes: int = 256
    use_compression: bool = False


# ------------------------------------------------------------------------------
#  Master Configuration
# ------------------------------------------------------------------------------
@dataclass
class AirBuddsConfig:
    """Top-level configuration container."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    ofdm: OFDMConfig = field(default_factory=OFDMConfig)
    qam: QAMConfig = field(default_factory=QAMConfig)
    preamble: PreambleConfig = field(default_factory=PreambleConfig)
    ultrasonic: UltrasonicConfig = field(default_factory=UltrasonicConfig)
    crc: CRCConfig = field(default_factory=CRCConfig)
    lms: LMSConfig = field(default_factory=LMSConfig)
    rir: RIRConfig = field(default_factory=RIRConfig)
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)

    project_name: str = "AirBudds"
    version: str = "1.0.0"
    team_name: str = "8J3D9"
    members: List[str] = field(
        default_factory=lambda: [
            "Joyanta Sutradhar (2305083)",
            "Dipbroto Karmokar Dip (2305089)",
        ]
    )

DEFAULT_CONFIG = AirBuddsConfig()
