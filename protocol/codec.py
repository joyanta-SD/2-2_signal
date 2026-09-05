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
