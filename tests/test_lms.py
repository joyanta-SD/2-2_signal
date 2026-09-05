from __future__ import annotations
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsp.lms_filter import LMSFilter
from config import LMSConfig

@pytest.fixture
def lms_filter():
    config = LMSConfig(num_taps=32, mu=0.01)
    return LMSFilter(config)

def test_lms_convergence(lms_filter):
    """Test LMS filter converges on sinusoidal noise: MSE decreases over time."""
    n = np.arange(1000)
    desired = np.sin(0.1 * n)
    reference = np.sin(0.1 * n + 0.5)
    
    output, error = lms_filter.filter_block(reference, desired)
    
    mse_start = np.mean(error[:200]**2)
    mse_end = np.mean(error[-200:]**2)
    assert mse_end < mse_start

def test_lms_noise_cancellation(lms_filter):
    """Test noise cancellation removes additive sine wave from signal."""
    n = np.arange(1000)
    clean_signal = np.random.normal(0, 0.1, 1000)
    noise_ref = np.sin(0.2 * n)
    noise_add = np.sin(0.2 * n + 0.3)
    
    desired = clean_signal + noise_add
    output, error = lms_filter.filter_block(noise_ref, desired)
    
    assert np.var(error[-200:]) < np.var(desired[-200:])

def test_lms_weights_converge(lms_filter):
    """Test filter weights converge to expected values for known system."""
    n = np.arange(1000)
    x = np.random.normal(0, 1, 1000)
    d = np.roll(x, 5) 
    d[:5] = 0
    
    lms_filter.filter_block(x, d)
    weights = lms_filter.get_weights()
    assert np.argmax(weights) == 5

def test_lms_learning_curve(lms_filter):
    """Test learning curve is monotonically decreasing (in moving average)."""
    n = np.arange(2000)
    x = np.random.normal(0, 1, 2000)
    d = np.roll(x, 2)
    _, error = lms_filter.filter_block(x, d)
    
    mse = [np.mean(error[i:i+100]**2) for i in range(0, 1900, 100)]
    assert mse[-1] < mse[0]

def test_lms_step_size():
    """Test with various step sizes (verify instability for large μ)."""
    n = np.arange(100)
    x = np.random.normal(0, 1, 100)
    d = np.roll(x, 1)
    
    config_stable = LMSConfig(num_taps=4, mu=0.01)
    f_stable = LMSFilter(config_stable)
    _, e_stable = f_stable.filter_block(x, d)
    assert not np.isnan(e_stable).any()
    
    config_unstable = LMSConfig(num_taps=4, mu=2.0)
    f_unstable = LMSFilter(config_unstable)
    _, e_unstable = f_unstable.filter_block(x, d)
    assert np.isnan(e_unstable).any() or np.max(np.abs(e_unstable)) > 1e3

def test_lms_reset(lms_filter):
    """Test reset() clears weights."""
    n = np.arange(100)
    x = np.random.normal(0, 1, 100)
    d = np.roll(x, 2)
    lms_filter.filter_block(x, d)
    assert np.sum(np.abs(lms_filter.get_weights())) > 0
    
    lms_filter.reset()
    assert np.sum(np.abs(lms_filter.get_weights())) == 0.0
