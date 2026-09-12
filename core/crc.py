from __future__ import annotations
import binascii
import numpy as np
import struct
from config import AirBuddsConfig, DEFAULT_CONFIG

class CRCEngine:
    """
    CRC-32 Engine for error detection in data packets.
    Uses standard IEEE 802.3 polynomial for robust error detection.
    """
    def __init__(self, config=None):
        """
        Initialize the CRC-32 Engine.
        """
        self.config = config if config else DEFAULT_CONFIG.crc
        self.poly = 0xEDB88320

    def compute(self, data: bytes) -> int:
        """
        Compute CRC-32 checksum for byte data.
        
        Parameters:
            data (bytes): Input byte string.
            
        Returns:
            int: 32-bit CRC checksum.
        """
        return binascii.crc32(data) & 0xFFFFFFFF

    def compute_bits(self, bits: np.ndarray) -> int:
        """
        Compute CRC from a bit array.
        
        Parameters:
            bits (np.ndarray): 1D array of bits.
            
        Returns:
            int: 32-bit CRC checksum.
        """
        data = np.packbits(bits).tobytes()
        return self.compute(data)

    def verify(self, data: bytes, checksum: int = None) -> bool:
        """
        Verify data against checksum.
        If checksum is None, assumes the last 4 bytes of data are the big-endian CRC.
        """
        if checksum is None:
            if len(data) < 4:
                return False
            checksum = int.from_bytes(data[-4:], byteorder='big')
            data = data[:-4]
        return self.compute(data) == checksum

    def append_checksum(self, data: bytes) -> bytes:
        """
        Append 4-byte big-endian CRC-32 to data.
        """
        crc = self.compute(data)
        return data + struct.pack('>I', crc)

    def verify_and_strip(self, data_with_crc: bytes) -> tuple[bytes, bool]:
        """
        Verify and strip CRC from packet.
        
        Returns:
            tuple[bytes, bool]: (data, is_valid)
        """
        if len(data_with_crc) < 4:
            return data_with_crc, False
        data = data_with_crc[:-4]
        expected_crc = struct.unpack('>I', data_with_crc[-4:])[0]
        is_valid = (self.compute(data) == expected_crc)
        return data, is_valid
