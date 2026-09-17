from __future__ import annotations
import numpy as np
import scipy.interpolate as interpolate
from config import AirBuddsConfig, DEFAULT_CONFIG

class ChannelEstimator:

    def __init__(self, config=None):

        self.config = config if config else DEFAULT_CONFIG.ofdm
        self.last_H = None
        self.last_H_interpolated = None

    def estimate(self, rx_pilots: np.ndarray, tx_pilots: np.ndarray, pilot_indices: np.ndarray, all_indices: Any) -> np.ndarray:

        if isinstance(all_indices, (int, np.integer)):
            all_indices = np.arange(int(all_indices))
        else:
            all_indices = np.asarray(all_indices)

        H_pilots = rx_pilots / tx_pilots

        H_all = self._interpolate(pilot_indices, H_pilots, all_indices)

        self.last_H = H_pilots
        self.last_H_interpolated = H_all

        return H_all

    def _interpolate(self, pilot_indices: np.ndarray, H_pilots: np.ndarray, all_indices: np.ndarray) -> np.ndarray:

        mags = np.abs(H_pilots)
        phases = np.unwrap(np.angle(H_pilots))

        f_mag = interpolate.interp1d(pilot_indices, mags, kind='linear', fill_value='extrapolate')
        f_phase = interpolate.interp1d(pilot_indices, phases, kind='linear', fill_value='extrapolate')

        H_mag = np.maximum(f_mag(all_indices), 1e-6)
        H_phase = f_phase(all_indices)

        return H_mag * np.exp(1j * H_phase)

    def equalize_zf(self, Y: np.ndarray, H: np.ndarray) -> np.ndarray:

        epsilon = 1e-10
        H_safe = np.where(np.abs(H) < epsilon, epsilon, H)
        return Y / H_safe

    def equalize(self, Y: np.ndarray, H: np.ndarray, method: str = 'zf', noise_var: float = 0.01) -> np.ndarray:

        if str(method).lower() == 'mmse':
            return self.equalize_mmse(Y, H, noise_var=noise_var)
        return self.equalize_zf(Y, H)

    def equalize_mmse(self, Y: np.ndarray, H: np.ndarray, noise_var: float = 0.01) -> np.ndarray:

        H_conj = np.conj(H)
        H_mag_sq = np.abs(H)**2

        weight = H_conj / (H_mag_sq + noise_var)
        return Y * weight

    def get_channel_response(self) -> tuple[np.ndarray, np.ndarray] | None:

        if self.last_H_interpolated is not None:
            return np.abs(self.last_H_interpolated), np.angle(self.last_H_interpolated)
        return None
