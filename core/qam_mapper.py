from __future__ import annotations
import numpy as np
from config import AirBuddsConfig, DEFAULT_CONFIG

class QAMMapper:
    """
    QAM Constellation Mapper for mapping bits to complex symbols and vice versa.
    Supports Gray-coded M-QAM (e.g., 4-QAM, 16-QAM, 64-QAM).
    """
    def __init__(self, config=None):
        """
        Initialize the QAM Mapper.
        
        Parameters:
            config: QAMConfig object. Uses DEFAULT_CONFIG.qam if None.
        """
        self.config = config if config else DEFAULT_CONFIG.qam
        self.M = getattr(self.config, 'order', getattr(self.config, 'M', 4))  # QAMConfig uses 'order'
        self.k = int(np.log2(self.M)) # Bits per symbol
        if 2**self.k != self.M:
            raise ValueError("M must be a power of 2.")
            
        self.constellation = self._generate_constellation()
        self._normalize_constellation()
        
        # Reverse mapping for demodulation
        self.points = np.array(list(self.constellation.values()))
        self.bit_tuples = list(self.constellation.keys())

    def _gray_code(self, n: int) -> int:
        """
        Compute Gray code for integer n.
        
        Parameters:
            n (int): Integer index.
            
        Returns:
            int: Gray-coded integer.
        """
        return n ^ (n >> 1)

    def _generate_constellation(self) -> dict[tuple[int, ...], complex]:
        """
        Create Gray-coded M-QAM constellation.
        
        Returns:
            dict: Mapping from bit tuples to complex constellation points.
        """
        sqrt_M = int(np.sqrt(self.M))
        if sqrt_M * sqrt_M != self.M:
            raise ValueError("M must be a perfect square for square QAM.")
            
        constellation = {}
        bits_per_dim = self.k // 2
        
        for i in range(sqrt_M):
            gray_i = self._gray_code(i)
            # Map index i to PAM symbol: -sqrt_M + 1 + 2*i
            i_val = -sqrt_M + 1 + 2*i
            for j in range(sqrt_M):
                gray_j = self._gray_code(j)
                q_val = -sqrt_M + 1 + 2*j
                
                # Combine Gray codes for bits
                # I-component gets upper bits, Q-component gets lower bits
                total_gray = (gray_i << bits_per_dim) | gray_j
                
                # Convert to tuple of bits
                bit_tuple = tuple((total_gray >> b) & 1 for b in range(self.k - 1, -1, -1))
                
                constellation[bit_tuple] = complex(i_val, q_val)
                
        return constellation

    def _normalize_constellation(self):
        """
        Scale the constellation so the average power is 1.
        """
        avg_power = np.mean([abs(c)**2 for c in self.constellation.values()])
        self.scale_factor = np.sqrt(avg_power)
        
        for k in self.constellation:
            self.constellation[k] /= self.scale_factor

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        Map bit groups to complex QAM symbols.
        
        Parameters:
            bits (np.ndarray): 1D array of bits (must be multiple of self.k).
            
        Returns:
            np.ndarray: 1D array of complex QAM symbols.
        """
        if len(bits) % self.k != 0:
            raise ValueError(f"Number of bits must be a multiple of {self.k}.")
            
        num_symbols = len(bits) // self.k
        symbols = np.zeros(num_symbols, dtype=complex)
        
        for i in range(num_symbols):
            b_tuple = tuple(bits[i*self.k : (i+1)*self.k])
            symbols[i] = self.constellation[b_tuple]
            
        return symbols

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """
        Hard-decision minimum-distance demapping of symbols to bits.
        
        Parameters:
            symbols (np.ndarray): 1D array of complex received symbols.
            
        Returns:
            np.ndarray: 1D array of demodulated bits.
        """
        # Broadcast and calculate distances to all constellation points
        symbols = np.asarray(symbols)
        diff = symbols[:, np.newaxis] - self.points[np.newaxis, :]
        dists = np.abs(diff)
        
        min_indices = np.argmin(dists, axis=1)
        
        bits = []
        for idx in min_indices:
            bits.extend(self.bit_tuples[idx])
            
        return np.array(bits, dtype=np.uint8)

    def get_constellation_points(self) -> np.ndarray:
        """
        Return all constellation points (normalized).
        
        Returns:
            np.ndarray: 1D array of complex points.
        """
        return self.points

    def get_ideal_grid(self) -> np.ndarray:
        """
        Return the ideal un-normalized grid positions for plotting.
        
        Returns:
            np.ndarray: Ideal grid points.
        """
        return self.points * self.scale_factor
