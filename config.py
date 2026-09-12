from dataclasses import dataclass, field

@dataclass
class AudioConfig:
    sample_rate: int = 44_100
    device_in: int | None = None
    device_out: int | None = None
    volume: float = 0.95
    lead_in_seconds: float = 0.25
    lead_out_seconds: float = 0.20

@dataclass
class AirBuddsConfig:
    version: str = "3.0-Morse"
    team_name: str = "8J3D9"
    members: tuple = ("Joyanta Sutradhar (2305083)", "Dipbroto Karmokar Dip (2305089)")
    
    audio: AudioConfig = field(default_factory=AudioConfig)
    
    # Morse parameters
    morse_freq: float = 800.0  # Hz
    morse_wpm: float = 20.0    # Words per minute

DEFAULT_CONFIG = AirBuddsConfig()
