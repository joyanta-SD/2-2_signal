from __future__ import annotations
import numpy as np
from config import AirBuddsConfig, DEFAULT_CONFIG

class QAMMapper:

    def __init__(self, config=None):

        self.config = config if config else DEFAULT_CONFIG.qam
        self.M = getattr(self.config, 'order', getattr(self.config, 'M', 4))  
        self.k = int(np.log2(self.M)) 
        if 2**self.k != self.M:
            raise ValueError("M must be a power of 2.")

        self.constellation = self._generate_constellation()
        self._normalize_constellation()

        self.points = np.array(list(self.constellation.values()))
        self.bit_tuples = list(self.constellation.keys())

    def _gray_code(self, n: int) -> int:

        return n ^ (n >> 1)

    def _generate_constellation(self) -> dict[tuple[int, ...], complex]:

        sqrt_M = int(np.sqrt(self.M))
        if sqrt_M * sqrt_M != self.M:
            raise ValueError("M must be a perfect square for square QAM.")

        constellation = {}
        bits_per_dim = self.k // 2

        for i in range(sqrt_M):
            gray_i = self._gray_code(i)

            i_val = -sqrt_M + 1 + 2*i
            for j in range(sqrt_M):
                gray_j = self._gray_code(j)
                q_val = -sqrt_M + 1 + 2*j

                total_gray = (gray_i << bits_per_dim) | gray_j

                bit_tuple = tuple((total_gray >> b) & 1 for b in range(self.k - 1, -1, -1))

                constellation[bit_tuple] = complex(i_val, q_val)

        return constellation

    def _normalize_constellation(self):

        avg_power = np.mean([abs(c)**2 for c in self.constellation.values()])
        self.scale_factor = np.sqrt(avg_power)

        for k in self.constellation:
            self.constellation[k] /= self.scale_factor

    def modulate(self, bits: np.ndarray) -> np.ndarray:

        if len(bits) % self.k != 0:
            raise ValueError(f"Number of bits must be a multiple of {self.k}.")

        num_symbols = len(bits) // self.k
        symbols = np.zeros(num_symbols, dtype=complex)

        for i in range(num_symbols):
            b_tuple = tuple(bits[i*self.k : (i+1)*self.k])
            symbols[i] = self.constellation[b_tuple]

        return symbols

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:

        symbols = np.asarray(symbols)
        diff = symbols[:, np.newaxis] - self.points[np.newaxis, :]
        dists = np.abs(diff)

        min_indices = np.argmin(dists, axis=1)

        bits = []
        for idx in min_indices:
            bits.extend(self.bit_tuples[idx])

        return np.array(bits, dtype=np.uint8)

    def get_constellation_points(self) -> np.ndarray:

        return self.points

    def get_ideal_grid(self) -> np.ndarray:

        return self.points * self.scale_factor
