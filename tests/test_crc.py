from __future__ import annotations
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.crc import CRCEngine
from config import AirBuddsConfig, DEFAULT_CONFIG

@pytest.fixture
def crc_engine():
    """Fixture for CRCEngine."""
    config = DEFAULT_CONFIG.crc
    return CRCEngine(config)

def test_crc_deterministic(crc_engine):
    """Test CRCEngine.compute() gives deterministic results."""
    data = b"AirBudds"
    crc1 = crc_engine.compute(data)
    crc2 = crc_engine.compute(data)
    assert crc1 == crc2

def test_crc_verify_valid(crc_engine):
    """Test CRCEngine.verify() returns True for valid data+checksum."""
    data = b"AirBudds"
    crc = crc_engine.compute(data)
    data_with_crc = data + crc.to_bytes(4, byteorder='big')
    assert crc_engine.verify(data_with_crc)

def test_crc_verify_corrupted(crc_engine):
    """Test CRCEngine.verify() returns False for corrupted data."""
    data = b"AirBudds"
    crc = crc_engine.compute(data)
    data_with_crc = bytearray(data + crc.to_bytes(4, byteorder='big'))
    data_with_crc[0] ^= 0x01  # Corrupt first byte
    assert not crc_engine.verify(bytes(data_with_crc))

def test_crc_round_trip(crc_engine):
    """Test CRCEngine.append_checksum() and verify_and_strip() round-trip."""
    data = b"Test Message"
    packet = crc_engine.append_checksum(data)
    extracted_data, valid = crc_engine.verify_and_strip(packet)
    assert valid
    assert extracted_data == data

def test_crc_compute_bits(crc_engine):
    """Test compute_bits() matches compute() for same data."""
    data = b"\xAA\x55"
    import numpy as np
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    crc_bytes = crc_engine.compute(data)
    crc_bits = crc_engine.compute_bits(bits)
    assert crc_bytes == crc_bits

def test_crc32_known_vector(crc_engine):
    """Test against known CRC-32 test vectors: CRC32('123456789') = 0xCBF43926."""
    data = b"123456789"
    crc = crc_engine.compute(data)
    assert crc == 0xCBF43926
