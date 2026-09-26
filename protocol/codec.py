from __future__ import annotations
import numpy as np
import zlib
import os

class Codec:

    @staticmethod
    def text_to_bits(text: str) -> np.ndarray:

        return Codec.bytes_to_bits(text.encode('utf-8'))

    @staticmethod
    def bits_to_text(bits: np.ndarray) -> str:

        return Codec.bits_to_bytes(bits).decode('utf-8', errors='ignore')

    @staticmethod
    def bytes_to_bits(data: bytes) -> np.ndarray:

        byte_array = np.frombuffer(data, dtype=np.uint8)
        return np.unpackbits(byte_array)

    @staticmethod
    def bits_to_bytes(bits: np.ndarray) -> bytes:

        byte_array = np.packbits(bits)
        return byte_array.tobytes()

    @staticmethod
    def file_to_bits(filepath: str) -> tuple[np.ndarray, dict]:

        with open(filepath, 'rb') as f:
            data = f.read()

        metadata = {
            'filename': os.path.basename(filepath),
            'size': len(data)
        }
        return Codec.bytes_to_bits(data), metadata

    @staticmethod
    def bits_to_file(bits: np.ndarray, metadata: dict, output_dir: str = '.') -> str:

        data = Codec.bits_to_bytes(bits)

        if 'size' in metadata and metadata['size'] < len(data):
            data = data[:metadata['size']]

        filename = metadata.get('filename', 'received_file.dat')
        out_path = os.path.join(output_dir, filename)

        with open(out_path, 'wb') as f:
            f.write(data)

        return out_path

    @staticmethod
    def compress(data: bytes) -> bytes:

        return zlib.compress(data)

    @staticmethod
    def decompress(data: bytes) -> bytes:

        try:
            return zlib.decompress(data)
        except zlib.error:

            return b""

    @staticmethod
    def pad_bits(bits: np.ndarray, block_size: int) -> np.ndarray:

        remainder = len(bits) % block_size
        if remainder == 0:
            return bits

        padding_len = block_size - remainder
        padding = np.zeros(padding_len, dtype=bits.dtype)
        return np.concatenate((bits, padding))

    @staticmethod
    def unpad_bits(bits: np.ndarray, original_length: int) -> np.ndarray:

        if original_length > len(bits):
            return bits
        return bits[:original_length]

    @staticmethod
    def fec_encode(data: bytes, nsym: int = 10) -> bytes:

        import reedsolo
        rs = reedsolo.RSCodec(nsym)

        return bytes(rs.encode(data))

    @staticmethod
    def fec_decode(data: bytes, nsym: int = 10) -> bytes:
        import reedsolo
        rs = reedsolo.RSCodec(nsym)
        try:
            decoded_msg, decoded_msg_ecc, err_pos = rs.decode(data)
            return bytes(decoded_msg)
        except reedsolo.ReedSolomonError:
            # Best effort extraction if FEC fails
            chunk_size = 255
            data_size = chunk_size - nsym
            result = bytearray()
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i+chunk_size]
                result.extend(chunk[:data_size])
            return bytes(result)

    @staticmethod
    def interleave(data: bytes, depth: int = 8) -> bytes:

        data_len = len(data)

        pad_len = (depth - (data_len % depth)) % depth
        padded_data = data + b'\x00' * pad_len

        rows = len(padded_data) // depth
        matrix = np.frombuffer(padded_data, dtype=np.uint8).reshape((rows, depth))
        interleaved = matrix.T.tobytes()

        header = np.array([data_len], dtype=np.uint32).tobytes()
        return header + interleaved

    @staticmethod
    def deinterleave(data: bytes, depth: int = 8) -> bytes:

        if len(data) < 4:
            return b""

        header = data[:4]
        interleaved = data[4:]

        data_len = np.frombuffer(header, dtype=np.uint32)[0]

        if len(interleaved) == 0 or len(interleaved) % depth != 0:
            return b"" 

        rows = len(interleaved) // depth

        matrix = np.frombuffer(interleaved, dtype=np.uint8).reshape((depth, rows))
        deinterleaved = matrix.T.tobytes()

        return deinterleaved[:data_len]

