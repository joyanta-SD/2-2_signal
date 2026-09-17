from __future__ import annotations
import struct
import zlib
from config import AirBuddsConfig, DEFAULT_CONFIG

class FrameBuilder:

    def __init__(self, config: AirBuddsConfig = None):

        self.config = config or DEFAULT_CONFIG
        self.max_payload = getattr(self.config.protocol, 'max_payload_bytes', 512) if hasattr(self.config, 'protocol') else 512

    def build_frame(self, payload: bytes, modulation_order: int = 4) -> bytes:

        header = self._build_header(len(payload), modulation_order)

        data_to_crc = header + payload
        crc = zlib.crc32(data_to_crc) & 0xFFFFFFFF
        crc_bytes = struct.pack(">I", crc)
        return data_to_crc + crc_bytes

    def parse_frame(self, frame_data: bytes) -> tuple[dict, bytes, bool]:

        if len(frame_data) < 8:
            return {}, b"", False

        header_bytes = frame_data[:4]
        header_dict = self._parse_header(header_bytes)

        expected_len = header_dict['payload_length']
        if len(frame_data) < 4 + expected_len + 4:

            return header_dict, b"", False

        payload = frame_data[4:4+expected_len]
        crc_received = struct.unpack(">I", frame_data[4+expected_len:4+expected_len+4])[0]

        crc_calculated = zlib.crc32(header_bytes + payload) & 0xFFFFFFFF
        crc_valid = (crc_calculated == crc_received)

        return header_dict, payload, crc_valid

    def _build_header(self, payload_length: int, modulation_order: int, frame_id: int = 0) -> bytes:

        flags = 0  

        combined_mod_flags = ((modulation_order & 0x0F) << 4) | (flags & 0x0F)

        return struct.pack(">H B B", payload_length, frame_id, combined_mod_flags)

    def _parse_header(self, header_bytes: bytes) -> dict:

        payload_length, frame_id, combined_mod_flags = struct.unpack(">H B B", header_bytes)

        modulation_order = (combined_mod_flags >> 4) & 0x0F
        flags = combined_mod_flags & 0x0F

        return {
            'payload_length': payload_length,
            'frame_id': frame_id,
            'modulation_order': modulation_order,
            'flags': flags
        }

    def segment_data(self, data: bytes) -> list[bytes]:

        return [data[i:i + self.max_payload] for i in range(0, len(data), self.max_payload)]

    def reassemble_data(self, frames: list[bytes]) -> bytes:

        return b"".join(frames)
