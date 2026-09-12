from __future__ import annotations
import numpy as np
from config import AirBuddsConfig, DEFAULT_CONFIG

class LMSFilter:
    """LMS Adaptive Noise Cancellation filter."""
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        lms_cfg = getattr(self.config, 'lms', self.config)
        self.mu = getattr(lms_cfg, 'step_size', getattr(lms_cfg, 'mu', 0.01))
        self.leakage = getattr(lms_cfg, 'leakage', 1.0)
        self.filter_length = getattr(lms_cfg, 'filter_order', getattr(lms_cfg, 'num_taps', 64))
        self.weights = np.zeros(self.filter_length)
        self.learning_curve = []
        self.noise_profile = None

    def reset(self):
        """Reset weights to zero."""
        self.weights = np.zeros(self.filter_length)
        self.learning_curve = []

    def filter_block(self, reference: np.ndarray, desired: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run LMS filtering on a block, return (output, error)."""
        output, error, _ = self.adapt(desired, reference)
        return output, error

    def adapt(self, desired: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run LMS algorithm, return (output, error, weights_final)."""
        n_samples = len(desired)
        output = np.zeros(n_samples)
        error = np.zeros(n_samples)
        self.learning_curve = np.zeros(n_samples)
        
        # Buffer for reference signal
        x_buf = np.zeros(self.filter_length)
        
        for n in range(n_samples):
            # Shift buffer
            x_buf[1:] = x_buf[:-1]
            x_buf[0] = reference[n]
            
            # y[n] = w^T * x
            y_n = np.dot(self.weights, x_buf)
            output[n] = y_n
            
            # e[n] = d[n] - y[n]
            e_n = desired[n] - y_n
            error[n] = e_n
            
            # w[n+1] = lambda * w[n] + 2 * mu * e[n] * x[n]
            self.weights = self.leakage * self.weights + 2 * self.mu * e_n * x_buf
            
            self.learning_curve[n] = e_n**2
            
        return output, error, self.weights

    def estimate_noise_profile(self, noise_only: np.ndarray):
        """Learn noise characteristics from a silent segment."""
        self.noise_profile = np.mean(noise_only**2)

    def cancel_noise(self, noisy_signal: np.ndarray, noise_ref: np.ndarray = None) -> np.ndarray:
        """Apply noise cancellation."""
        if noise_ref is None:
            # Generate dummy reference if not provided
            var = np.sqrt(self.noise_profile) if self.noise_profile else 0.1
            noise_ref = np.random.normal(0, var, len(noisy_signal))
            
        self._step_size_check(noise_ref)
        _, error, _ = self.adapt(noisy_signal, noise_ref)
        return error

    def get_learning_curve(self) -> np.ndarray:
        """Return MSE over iterations for convergence plot."""
        return np.array(self.learning_curve)

    def get_weights(self) -> np.ndarray:
        """Return current filter weights."""
        return np.copy(self.weights)

    def _step_size_check(self, reference: np.ndarray):
        """Verify step size for stability."""
        var_x = np.var(reference)
        max_mu = 1.0 / (self.filter_length * var_x + 1e-12)
        if self.mu >= max_mu:
            self.mu = max_mu * 0.1
