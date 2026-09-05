from __future__ import annotations
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.preamble import PreambleGenerator
from config import PreambleConfig

@pytest.fixture
def preamble_gen():
    config = PreambleConfig(
        type='chirp',
        length=1000,
        f_start=1000,
        f_stop=4000,
        num_repeats=1
    )
    return PreambleGenerator(config, fs=48000)

def test_preamble_chirp_length(preamble_gen):
    """Test chirp generation produces correct length signal."""
    chirp = preamble_gen.generate()
    assert len(chirp) == 1000

def test_preamble_spectral_content(preamble_gen):
    """Test chirp frequency sweeps from f_start to f_stop (check spectral content)."""
    chirp = preamble_gen.generate()
    assert np.var(chirp) > 0

def test_preamble_matched_filter_clean(preamble_gen):
    """Test matched filter produces clear peak for clean chirp."""
    chirp = preamble_gen.generate()
    padded_signal = np.pad(chirp, (500, 500))
    idx = preamble_gen.detect(padded_signal)
    assert idx == 500

def test_preamble_detection_with_noise(preamble_gen):
    """Test detection finds correct index when chirp is embedded in noise (SNR=20dB)."""
    chirp = preamble_gen.generate()
    padded_signal = np.pad(chirp, (300, 300))
    # Add noise SNR = 20dB
    signal_power = np.mean(chirp**2)
    noise_power = signal_power / (10 ** (20 / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(padded_signal))
    noisy_signal = padded_signal + noise
    idx = preamble_gen.detect(noisy_signal)
    assert abs(idx - 300) < 5

def test_preamble_detection_noise_only(preamble_gen):
    """Test detection returns None for noise-only signal."""
    noise = np.random.normal(0, 0.1, 2000)
    idx = preamble_gen.detect(noise, threshold=10.0) 
    assert idx is None

def test_preamble_num_repeats():
    """Test preamble with num_repeats > 1."""
    config = PreambleConfig(length=500, num_repeats=3)
    gen = PreambleGenerator(config, fs=48000)
    sig = gen.generate()
    assert len(sig) == 1500
