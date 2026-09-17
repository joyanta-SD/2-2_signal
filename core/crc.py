from __future__ import annotations
import binascii
import numpy as np
import struct
from config import AirBuddsConfig, DEFAULT_CONFIG

class CRCEngine:

    def __init__(self, config=None):

        self.config = config if config else DEFAULT_CONFIG.crc
        self.poly = 0xEDB88320

    def compute(self, data: bytes) -> int:

        return binascii.crc32(data) & 0xFFFFFFFF

    def compute_bits(self, bits: np.ndarray) -> int:

        data = np.packbits(bits).tobytes()
        return self.compute(data)

    def verify(self, data: bytes, checksum: int = None) -> bool:

        if checksum is None:
            if len(data) < 4:
                return False
            checksum = int.from_bytes(data[-4:], byteorder='big')
            data = data[:-4]
        return self.compute(data) == checksum

    def append_checksum(self, data: bytes) -> bytes:

        crc = self.compute(data)
        return data + struct.pack('>I', crc)

    def verify_and_strip(self, data_with_crc: bytes) -> tuple[bytes, bool]:

        if len(data_with_crc) < 4:
            return data_with_crc, False
        data = data_with_crc[:-4]
        expected_crc = struct.unpack('>I', data_with_crc[-4:])[0]
        is_valid = (self.compute(data) == expected_crc)
        return data, is_valid
