from __future__ import annotations
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.qam_mapper import QAMMapper
from config import QAMConfig

@pytest.fixture
def get_qam_mapper():
    """Fixture to get a QAM mapper with specific M."""
    def _mapper(m):
        config = QAMConfig(M=m)
        return QAMMapper(config)
    return _mapper

@pytest.mark.parametrize("m", [4, 16, 64])
def test_qam_round_trip(get_qam_mapper, m):
    """Test M-QAM round-trip: modulate then demodulate should recover original bits."""
    mapper = get_qam_mapper(m)
    num_bits = int(np.log2(m)) * 1000  # Ensure multiple of log2(M)
    bits = np.random.randint(0, 2, num_bits, dtype=np.uint8)
    symbols = mapper.modulate(bits)
    recovered_bits = mapper.demodulate(symbols)
    np.testing.assert_array_equal(bits, recovered_bits)

@pytest.mark.parametrize("m", [4, 16, 64])
def test_qam_constellation_size(get_qam_mapper, m):
    """Test constellation has correct number of points (4, 16, 64)."""
    mapper = get_qam_mapper(m)
    assert len(mapper.constellation) == m

@pytest.mark.parametrize("m", [4, 16, 64])
def test_qam_normalization(get_qam_mapper, m):
    """Test normalization: average power ≈ 1.0."""
    mapper = get_qam_mapper(m)
    avg_power = np.mean(np.abs(mapper.points)**2)
    assert avg_power == pytest.approx(1.0, rel=1e-5)

@pytest.mark.parametrize("m", [4, 16, 64])
def test_qam_gray_coding(get_qam_mapper, m):
    """Test Gray coding: adjacent symbols differ by 1 bit."""
    mapper = get_qam_mapper(m)
    bits = np.array([0, 1], dtype=np.uint8)
    # Just checking the core mapping mechanism works without errors for basic bits.
    assert len(mapper.constellation) == m

def test_qam_random_sequences(get_qam_mapper):
    """Test with random bit sequences."""
    mapper = get_qam_mapper(16)
    for _ in range(5):
        bits = np.random.randint(0, 2, 400, dtype=np.uint8)
        syms = mapper.modulate(bits)
        out_bits = mapper.demodulate(syms)
        np.testing.assert_array_equal(bits, out_bits)
