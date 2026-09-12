from __future__ import annotations
import numpy as np
import zlib
import os

class Codec:
    """
    Binary Codec class for data transformations.
    Handles conversions between text/bytes/files and bit arrays, plus compression and padding.
    """
    
    @staticmethod
    def text_to_bits(text: str) -> np.ndarray:
        """
        Convert UTF-8 string to a bit array.
        
        Parameters:
            text: Input string.
            
        Returns:
            np.ndarray: Array of bits (0s and 1s) as uint8.
        """
        return Codec.bytes_to_bits(text.encode('utf-8'))

    @staticmethod
    def bits_to_text(bits: np.ndarray) -> str:
        """
        Convert bit array to UTF-8 text.
        
        Parameters:
            bits: Input bit array.
            
        Returns:
            str: Decoded string.
        """
        return Codec.bits_to_bytes(bits).decode('utf-8', errors='ignore')

    @staticmethod
    def bytes_to_bits(data: bytes) -> np.ndarray:
        """
        Convert bytes to a numpy bit array.
        
        Parameters:
            data: Input byte sequence.
            
        Returns:
            np.ndarray: Array of bits (0s and 1s) as uint8.
        """
        # Convert bytes to numpy uint8 array, then unpack bits (MSB first)
        byte_array = np.frombuffer(data, dtype=np.uint8)
        return np.unpackbits(byte_array)

    @staticmethod
    def bits_to_bytes(bits: np.ndarray) -> bytes:
        """
        Convert numpy bit array to bytes.
        
        Parameters:
            bits: Input bit array.
            
        Returns:
            bytes: Reconstructed byte sequence.
        """
        # Pack bits into uint8 array (MSB first), then convert to bytes
        byte_array = np.packbits(bits)
        return byte_array.tobytes()

    @staticmethod
    def file_to_bits(filepath: str) -> tuple[np.ndarray, dict]:
        """
        Read a file and convert its contents to a bit array.
        
        Parameters:
            filepath: Path to the input file.
            
        Returns:
            tuple: (bit_array, metadata_dict)
        """
        with open(filepath, 'rb') as f:
            data = f.read()
            
        metadata = {
            'filename': os.path.basename(filepath),
            'size': len(data)
        }
        return Codec.bytes_to_bits(data), metadata

    @staticmethod
    def bits_to_file(bits: np.ndarray, metadata: dict, output_dir: str = '.') -> str:
        """
        Reconstruct a file from a bit array and save it.
        
        Parameters:
            bits: The bit array of the file contents.
            metadata: Metadata dict containing 'filename'.
            output_dir: Directory to save the output file.
            
        Returns:
            str: The output file path.
        """
        data = Codec.bits_to_bytes(bits)
        
        # Ensure correct size if padded
        if 'size' in metadata and metadata['size'] < len(data):
            data = data[:metadata['size']]
            
        filename = metadata.get('filename', 'received_file.dat')
        out_path = os.path.join(output_dir, filename)
        
        with open(out_path, 'wb') as f:
            f.write(data)
            
        return out_path

    @staticmethod
    def compress(data: bytes) -> bytes:
        """
        Compress byte data using zlib.
        
        Parameters:
            data: Uncompressed bytes.
            
        Returns:
            bytes: Compressed bytes.
        """
        return zlib.compress(data)

    @staticmethod
    def decompress(data: bytes) -> bytes:
        """
        Decompress byte data using zlib.
        
        Parameters:
            data: Compressed bytes.
            
        Returns:
            bytes: Decompressed bytes.
        """
        try:
            return zlib.decompress(data)
        except zlib.error:
            # Fallback if corrupted
            return b""

    @staticmethod
    def pad_bits(bits: np.ndarray, block_size: int) -> np.ndarray:
        """
        Zero-pad a bit array so its length is a multiple of block_size.
        
        Parameters:
            bits: Input bit array.
            block_size: The target block size.
            
        Returns:
            np.ndarray: Padded bit array.
        """
        remainder = len(bits) % block_size
        if remainder == 0:
            return bits
            
        padding_len = block_size - remainder
        padding = np.zeros(padding_len, dtype=bits.dtype)
        return np.concatenate((bits, padding))

    @staticmethod
    def unpad_bits(bits: np.ndarray, original_length: int) -> np.ndarray:
        """
        Remove padding from a bit array.
        
        Parameters:
            bits: The padded bit array.
            original_length: The expected original length of the bit array.
            
        Returns:
            np.ndarray: Truncated bit array.
        """
        if original_length > len(bits):
            return bits
        return bits[:original_length]

    @staticmethod
    def fec_encode(data: bytes, nsym: int = 10) -> bytes:
        """
        Encode bytes using Reed-Solomon Forward Error Correction.
        
        Parameters:
            data: Input byte sequence.
            nsym: Number of ECC symbols to add per chunk.
            
        Returns:
            bytes: FEC encoded byte sequence.
        """
        import reedsolo
        rs = reedsolo.RSCodec(nsym)
        # reedsolo processes in chunks of 255-nsym bytes max.
        # rs.encode handles chunking internally if we pass bytearray/bytes
        return bytes(rs.encode(data))

    @staticmethod
    def fec_decode(data: bytes, nsym: int = 10) -> bytes:
        """
        Decode bytes using Reed-Solomon Forward Error Correction.
        
        Parameters:
            data: FEC encoded byte sequence.
            nsym: Number of ECC symbols that were added.
            
        Returns:
            bytes: Decoded (error-corrected) byte sequence.
        """
        import reedsolo
        rs = reedsolo.RSCodec(nsym)
        try:
            decoded_msg, decoded_msg_ecc, err_pos = rs.decode(data)
            return bytes(decoded_msg)
        except reedsolo.ReedSolomonError:
            # If correction fails, return original data stripped of ECC as best effort
            # (or raise an exception for upper layers to handle)
            raise ValueError("FEC Decoding failed: Too many errors")

    @staticmethod
    def interleave(data: bytes, depth: int = 8) -> bytes:
        """
        Block byte interleaver to spread burst errors across multiple RS blocks.
        
        Parameters:
            data: Input byte sequence.
            depth: Number of columns in the interleaver matrix.
            
        Returns:
            bytes: Interleaved byte sequence.
        """
        data_len = len(data)
        # Pad data so it's a multiple of depth
        pad_len = (depth - (data_len % depth)) % depth
        padded_data = data + b'\x00' * pad_len
        
        # Reshape into (N, depth) and transpose to (depth, N)
        rows = len(padded_data) // depth
        matrix = np.frombuffer(padded_data, dtype=np.uint8).reshape((rows, depth))
        interleaved = matrix.T.tobytes()
        
        # We need to prepend the original length to correctly deinterleave,
        # but let's just use a fixed 4-byte header for the original length
        header = np.array([data_len], dtype=np.uint32).tobytes()
        return header + interleaved

    @staticmethod
    def deinterleave(data: bytes, depth: int = 8) -> bytes:
        """
        Block byte deinterleaver.
        
        Parameters:
            data: Interleaved byte sequence (must include header from interleave).
            depth: Number of columns used in interleaver.
            
        Returns:
            bytes: Deinterleaved byte sequence.
        """
        if len(data) < 4:
            return b""
            
        header = data[:4]
        interleaved = data[4:]
        
        data_len = np.frombuffer(header, dtype=np.uint32)[0]
        
        if len(interleaved) == 0 or len(interleaved) % depth != 0:
            return b"" # Corrupted
            
        rows = len(interleaved) // depth
        # Reshape (depth, N) back to (N, depth)
        matrix = np.frombuffer(interleaved, dtype=np.uint8).reshape((depth, rows))
        deinterleaved = matrix.T.tobytes()
        
        return deinterleaved[:data_len]

