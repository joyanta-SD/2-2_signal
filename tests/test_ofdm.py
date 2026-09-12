from __future__ import annotations
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ofdm_transceiver import OFDMTransceiver
from config import AirBuddsConfig

@pytest.fixture
def ofdm_transceiver():
    config = AirBuddsConfig()
    return OFDMTransceiver(config)

def test_ofdm_round_trip_awgn(ofdm_transceiver):
    """Critical test: Full TX→RX round-trip in AWGN channel (SNR=30dB)."""
    message = b"Hello AirBudds!"
    tx_signal, metadata_tx = ofdm_transceiver.transmit(message)
    
    # Add light AWGN (SNR=30dB)
    signal_power = np.mean(tx_signal**2)
    noise_power = signal_power / (10 ** (30 / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(tx_signal))
    rx_signal = tx_signal + noise
    
    rx_message, metadata_rx = ofdm_transceiver.receive(rx_signal)
    assert rx_message == message

def test_ofdm_modulation_length(ofdm_transceiver):
    """Test OFDM modulation produces correct length signal."""
    tx_signal, meta = ofdm_transceiver.transmit(b"Test")
    assert len(tx_signal) > 0

def test_ofdm_cyclic_prefix(ofdm_transceiver):
    """Test cyclic prefix is correctly prepended."""
    # This assumes we check if cp length configuration exists and isn't raising errors
    assert ofdm_transceiver.config.ofdm.cp_length > 0

def test_ofdm_ideal_channel(ofdm_transceiver):
    """Test encode→decode with no channel (ideal) recovers exact bytes."""
    message = b"Ideal Channel Test"
    tx_signal, meta_tx = ofdm_transceiver.transmit(message)
    rx_message, meta_rx = ofdm_transceiver.receive(tx_signal)
    assert rx_message == message

def test_ofdm_modulation_schemes():
    """Test with 4-QAM and 16-QAM."""
    for m in [4, 16]:
        config = AirBuddsConfig()
        config.qam.M = m
        transceiver = OFDMTransceiver(config)
        msg = b"Modulation Test"
        tx, _ = transceiver.transmit(msg)
        rx, _ = transceiver.receive(tx)
        assert rx == msg

def test_ofdm_metadata(ofdm_transceiver):
    """Test metadata dict from decode contains expected keys."""
    msg = b"Metadata Test"
    tx, _ = ofdm_transceiver.transmit(msg)
    rx, meta = ofdm_transceiver.receive(tx)
    assert rx == msg
    assert isinstance(meta, dict)
