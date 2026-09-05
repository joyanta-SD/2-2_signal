from __future__ import annotations
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.channel_estimator import ChannelEstimator
from config import OFDMConfig

@pytest.fixture
def channel_estimator():
    config = OFDMConfig()
    return ChannelEstimator(config)

def test_channel_estimation_flat(channel_estimator):
    """Test channel estimation with known flat channel H[k]=1."""
    rx_pilots = np.ones(10, dtype=complex)
    tx_pilots = np.ones(10, dtype=complex)
    pilot_indices = np.linspace(0, 63, 10, dtype=int)
    H_est = channel_estimator.estimate(rx_pilots, tx_pilots, pilot_indices, 64)
    np.testing.assert_allclose(H_est, np.ones(64), atol=1e-5)

def test_channel_zero_forcing(channel_estimator):
    """Test Zero-Forcing equalization recovers transmitted symbols."""
    rx_symbols = np.array([1.0, 2.0, -1.0]) * 2.0
    H = np.array([2.0, 2.0, 2.0])
    eq_symbols = channel_estimator.equalize(rx_symbols, H, method='zf')
    np.testing.assert_allclose(eq_symbols, np.array([1.0, 2.0, -1.0]), atol=1e-5)

def test_channel_mmse(channel_estimator):
    """Test MMSE equalization with noisy channel."""
    rx_symbols = np.array([1.0, 2.0, -1.0]) * 2.0
    H = np.array([2.0, 2.0, 2.0])
    eq_symbols = channel_estimator.equalize(rx_symbols, H, method='mmse', noise_var=0.001)
    np.testing.assert_allclose(eq_symbols, np.array([1.0, 2.0, -1.0]), atol=1e-2)

def test_channel_interpolation(channel_estimator):
    """Test interpolation between pilot subcarriers."""
    rx_pilots = np.array([1.0, 2.0, 3.0])
    tx_pilots = np.array([1.0, 1.0, 1.0])
    pilot_indices = np.array([0, 32, 63])
    H_est = channel_estimator.estimate(rx_pilots, tx_pilots, pilot_indices, 64)
    assert H_est[0] == pytest.approx(1.0)
    assert H_est[32] == pytest.approx(2.0)
    assert H_est[63] == pytest.approx(3.0)

def test_channel_frequency_selective(channel_estimator):
    """Test with frequency-selective channel (varying H[k])."""
    pilot_indices = np.arange(0, 64, 4)
    tx_pilots = np.ones(len(pilot_indices))
    true_H = np.exp(-1j * np.arange(64) * 0.1)
    rx_pilots = tx_pilots * true_H[pilot_indices]
    
    H_est = channel_estimator.estimate(rx_pilots, tx_pilots, pilot_indices, 64)
    np.testing.assert_allclose(H_est[pilot_indices], true_H[pilot_indices], atol=1e-2)
