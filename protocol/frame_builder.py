from __future__ import annotations
import struct
import zlib
from config import AirBuddsConfig, DEFAULT_CONFIG

class FrameBuilder:
    """
    Packet Framing class for constructing and parsing data frames.
    Format: [Header (4 bytes) | Payload | CRC-32 (4 bytes)]
    Header format: [payload_length (16 bits) | modulation_order (4 bits) | frame_id (8 bits) | flags (4 bits)]
    """
    def __init__(self, config: AirBuddsConfig = None):
        """
        Initialize FrameBuilder.
        
        Parameters:
            config: AirBudds configuration.
        """
        self.config = config or DEFAULT_CONFIG
        self.max_payload = getattr(self.config.protocol, 'max_payload_bytes', 512) if hasattr(self.config, 'protocol') else 512
        
    def build_frame(self, payload: bytes, modulation_order: int = 4) -> bytes:
        """
        Construct a complete frame with header and CRC-32.
        
        Parameters:
            payload: Data bytes to transmit.
            modulation_order: The modulation order used (e.g., 4 for QPSK).
            
        Returns:
            bytes: The assembled frame.
        """
        header = self._build_header(len(payload), modulation_order)
        # Compute CRC on header + payload
        data_to_crc = header + payload
        crc = zlib.crc32(data_to_crc) & 0xFFFFFFFF
        crc_bytes = struct.pack(">I", crc)
        return data_to_crc + crc_bytes
        
    def parse_frame(self, frame_data: bytes) -> tuple[dict, bytes, bool]:
        """
        Parse a received frame.
        
        Parameters:
            frame_data: The raw byte data of the received frame.
            
        Returns:
            tuple: (header_dict, payload_bytes, crc_valid_bool)
        """
        if len(frame_data) < 8:
            return {}, b"", False
            
        header_bytes = frame_data[:4]
        header_dict = self._parse_header(header_bytes)
        
        expected_len = header_dict['payload_length']
        if len(frame_data) < 4 + expected_len + 4:
            # Frame is too short
            return header_dict, b"", False
            
        payload = frame_data[4:4+expected_len]
        crc_received = struct.unpack(">I", frame_data[4+expected_len:4+expected_len+4])[0]
        
        crc_calculated = zlib.crc32(header_bytes + payload) & 0xFFFFFFFF
        crc_valid = (crc_calculated == crc_received)
        
        return header_dict, payload, crc_valid
        
    def _build_header(self, payload_length: int, modulation_order: int, frame_id: int = 0) -> bytes:
        """
        Encode metadata into a 4-byte header.
        
        Parameters:
            payload_length: Length of the payload in bytes (up to 65535).
            modulation_order: Modulation order (up to 15).
            frame_id: Sequence ID of the frame (0-255).
            
        Returns:
            bytes: 4-byte header.
        """
        flags = 0  # Reserved for future use
        
        # Packing:
        # byte 0-1: payload_length (16 bits)
        # byte 2: frame_id (8 bits)
        # byte 3: modulation_order (4 bits) | flags (4 bits)
        
        combined_mod_flags = ((modulation_order & 0x0F) << 4) | (flags & 0x0F)
        
        return struct.pack(">H B B", payload_length, frame_id, combined_mod_flags)
        
    def _parse_header(self, header_bytes: bytes) -> dict:
        """
        Decode a 4-byte header.
        
        Parameters:
            header_bytes: 4 bytes of header data.
            
        Returns:
            dict: Parsed header metadata.
        """
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
        """
        Split large data into chunks of max_payload_bytes.
        
        Parameters:
            data: Raw data to be segmented.
            
        Returns:
            list[bytes]: List of payload segments.
        """
        return [data[i:i + self.max_payload] for i in range(0, len(data), self.max_payload)]
        
    def reassemble_data(self, frames: list[bytes]) -> bytes:
        """
        Reassemble payload from multiple frames.
        
        Parameters:
            frames: List of verified payload byte segments.
            
        Returns:
            bytes: The complete reassembled data.
        """
        return b"".join(frames)
