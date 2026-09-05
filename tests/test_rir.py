from __future__ import annotations
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsp.rir_simulator import RIRSimulator
from config import RIRConfig

def test_rir_anechoic():
    """Test anechoic preset produces impulse response ≈ delta function."""
    config = RIRConfig(preset='anechoic')
    sim = RIRSimulator(config)
    h = sim.get_impulse_response()
    assert np.max(np.abs(h)) == pytest.approx(1.0)
    assert np.sum(np.abs(h) > 0.01) == 1

def test_rir_small_room():
    """Test small_room preset produces multiple reflections."""
    config = RIRConfig(preset='small_room')
    sim = RIRSimulator(config)
    h = sim.get_impulse_response()
    assert len(h) > 1
    assert np.sum(np.abs(h) > 0.01) > 1

def test_rir_apply_channel():
    """Test apply_channel output length."""
    config = RIRConfig(preset='anechoic')
    sim = RIRSimulator(config)
    signal = np.ones(100)
    out = sim.apply_channel(signal)
    h = sim.get_impulse_response()
    expected_len = len(signal) + len(h) - 1
    assert len(out) in [expected_len, len(signal)]

def test_rir_add_noise():
    """Test add_noise achieves target SNR within ±2 dB."""
    config = RIRConfig()
    sim = RIRSimulator(config)
    signal = np.random.normal(0, 1, 10000)
    snr_target = 20
    noisy = sim.add_noise(signal, snr_db=snr_target)
    
    noise = noisy - signal
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    snr_est = 10 * np.log10(signal_power / noise_power)
    
    assert abs(snr_est - snr_target) < 2.0

def test_rir_simulate():
    """Test simulate combines channel and noise."""
    config = RIRConfig(preset='small_room', snr_db=15)
    sim = RIRSimulator(config)
    signal = np.ones(500)
    out = sim.simulate(signal)
    assert len(out) >= len(signal)

def test_rir_presets_valid():
    """Test each preset generates valid impulse response."""
    for preset in ['anechoic', 'small_room', 'large_room']:
        config = RIRConfig(preset=preset)
        sim = RIRSimulator(config)
        h = sim.get_impulse_response()
        assert len(h) > 0
