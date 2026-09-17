from __future__ import annotations
import numpy as np
from config import AirBuddsConfig, DEFAULT_CONFIG
from .qam_mapper import QAMMapper
from .preamble import PreambleGenerator
from .channel_estimator import ChannelEstimator
from .synchronizer import Synchronizer
from .crc import CRCEngine
from dsp.filters import BandpassFilter

class OFDMTransceiver:

    def __init__(self, config: AirBuddsConfig = None):

        self.config = config if config else DEFAULT_CONFIG

        self.qam = QAMMapper(self.config.qam)
        self.sample_rate = getattr(self.config.audio, 'sample_rate', 44100)
        self.preamble = PreambleGenerator(self.config.preamble, self.sample_rate)
        self.channel_est = ChannelEstimator(self.config.ofdm)
        self.sync = Synchronizer(self.config)
        self.crc = CRCEngine(self.config.crc)

        self.n_fft = getattr(self.config.ofdm, 'fft_size',
                             getattr(self.config.ofdm, 'n_fft', 1024))
        self.cp_length = getattr(self.config.ofdm, 'cp_length', 512) 

        low_f = self.config.ofdm.guard_band_left * self.config.ofdm.subcarrier_spacing
        high_f = (self.n_fft // 2 - self.config.ofdm.guard_band_right) * self.config.ofdm.subcarrier_spacing
        self.bp_filter = BandpassFilter(low_freq=low_f, high_freq=high_f, sample_rate=self.sample_rate, order=5, method='butter')

        try:
            self.pilot_carriers = self.config.ofdm.pilot_indices()
            self.data_carriers = self.config.ofdm.data_indices()
        except (AttributeError, TypeError):
            self.pilot_carriers = getattr(self.config.ofdm, 'pilot_carriers',
                                          np.arange(10, 500, 10))
            all_active = getattr(self.config.ofdm, 'active_carriers',
                                 np.arange(1, 512))
            self.data_carriers = np.setdiff1d(all_active, self.pilot_carriers)

        np.random.seed(42)
        self.pilot_symbols = np.random.choice([1, -1], size=len(self.pilot_carriers))
        self.symbol_repeats = getattr(self.config.ofdm, 'symbol_repeats', 2)

        self._last_tx_signal = None
        self._last_rx_constellation_pre = None
        self._last_rx_constellation_post = None

    def encode(self, data: bytes) -> np.ndarray:

        data_with_crc = self.crc.append_checksum(data)

        from protocol.codec import Codec
        fec_encoded = Codec.fec_encode(data_with_crc, nsym=30)
        interleaved = Codec.interleave(fec_encoded, depth=16)

        import struct
        framed_payload = struct.pack('>H', len(interleaved)) + interleaved

        bits = np.unpackbits(np.frombuffer(framed_payload, dtype=np.uint8))

        k = self.qam.k
        capacity_per_symbol = (len(self.data_carriers) // 2) * k
        pad_len = (capacity_per_symbol - (len(bits) % capacity_per_symbol)) % capacity_per_symbol
        if pad_len > 0:
            bits = np.pad(bits, (0, pad_len), 'constant')

        unique_qam_symbols = self._bits_to_qam(bits)

        syms_per_ofdm = len(self.data_carriers) // 2
        num_ofdm_symbols = len(unique_qam_symbols) // syms_per_ofdm

        ofdm_blocks = []
        for i in range(num_ofdm_symbols):
            block_data = unique_qam_symbols[i * syms_per_ofdm : (i + 1) * syms_per_ofdm]

            full_block_data = np.concatenate([block_data, block_data])
            subcarriers = self._insert_pilots(full_block_data)
            for _ in range(self.symbol_repeats):
                ofdm_blocks.append(subcarriers)

        ofdm_signal = self._ofdm_modulate(ofdm_blocks)

        max_ofdm = np.max(np.abs(ofdm_signal))
        if max_ofdm > 0:
            ofdm_signal = (ofdm_signal / max_ofdm) * 0.95

        preamble_sig = self.preamble.preamble
        max_pre = np.max(np.abs(preamble_sig))
        if max_pre > 0:
            preamble_sig = (preamble_sig / max_pre) * 0.95

        tx_signal = np.concatenate([preamble_sig, ofdm_signal]).astype(np.float32)

        self._last_tx_signal = tx_signal
        return tx_signal

    def _bits_to_qam(self, bits: np.ndarray) -> np.ndarray:
        return self.qam.modulate(bits)

    def _insert_pilots(self, data_symbols: np.ndarray) -> np.ndarray:

        subcarriers = np.zeros(self.n_fft, dtype=complex)
        subcarriers[self.pilot_carriers] = self.pilot_symbols
        subcarriers[self.data_carriers] = data_symbols

        half = self.n_fft // 2
        subcarriers[half + 1:] = np.conj(subcarriers[1:half][::-1])
        subcarriers[0] = 0.0      
        subcarriers[half] = 0.0   
        return subcarriers

    def _ofdm_modulate(self, subcarrier_blocks: list[np.ndarray]) -> np.ndarray:
        time_blocks = []
        for block in subcarrier_blocks:

            time_sym = np.real(np.fft.ifft(block, n=self.n_fft))

            time_sym_cp = self._add_cyclic_prefix(time_sym)
            time_blocks.append(time_sym_cp)
        return np.concatenate(time_blocks).astype(np.float32)

    def _add_cyclic_prefix(self, symbol: np.ndarray) -> np.ndarray:
        return np.concatenate([symbol[-self.cp_length:], symbol])

    def decode(self, rx_signal: np.ndarray) -> tuple[bytes, dict]:

        meta = {
            'crc_valid': False,
            'num_ofdm_symbols': 0,
            'sync_offset': 0,
            'cfo_estimate': 0.0
        }

        if rx_signal.ndim > 1:
            rx_signal = np.mean(rx_signal, axis=-1)
        rx_signal = np.asarray(rx_signal, dtype=np.float32).flatten()

        max_rx = np.max(np.abs(rx_signal)) if len(rx_signal) > 0 else 0
        if max_rx > 1e-6:
            rx_signal = (rx_signal / max_rx) * 0.95

        rx_signal = self.bp_filter.apply(rx_signal)

        corrected_sig, start_idx, cfo = self.sync.synchronize(rx_signal)
        meta['sync_offset'] = start_idx
        meta['cfo_estimate'] = cfo

        payload_signal = corrected_sig[start_idx:]

        freq_blocks = self._ofdm_demodulate(payload_signal)
        meta['num_ofdm_symbols'] = len(freq_blocks)
        if len(freq_blocks) == 0:
            return b'', meta

        if self.symbol_repeats > 1 and len(freq_blocks) >= self.symbol_repeats:
            averaged_blocks = []
            for i in range(0, len(freq_blocks) - (len(freq_blocks) % self.symbol_repeats), self.symbol_repeats):
                group = freq_blocks[i : i + self.symbol_repeats]
                averaged_blocks.append(np.mean(group, axis=0))
            freq_blocks = averaged_blocks

        all_rx_data = []
        pre_eq_syms = []
        post_eq_syms = []

        for block in freq_blocks:
            rx_pilots, rx_data, p_idx, d_idx = self._extract_pilots_and_data(block)
            pre_eq_syms.extend(rx_data)

            H = self.channel_est.estimate(rx_pilots, self.pilot_symbols, p_idx, d_idx)

            eq_data = self.channel_est.equalize_zf(rx_data, H)

            half = len(eq_data) // 2
            combined_data = (eq_data[:half] + eq_data[half:]) / 2.0

            post_eq_syms.extend(combined_data)
            all_rx_data.extend(combined_data)

        self._last_rx_constellation_pre = np.array(pre_eq_syms)
        self._last_rx_constellation_post = np.array(post_eq_syms)

        bits = self.qam.demodulate(np.array(all_rx_data))

        num_bytes = len(bits) // 8
        bits = bits[:num_bytes * 8]
        decoded_bytes = np.packbits(bits).tobytes()

        import struct
        from protocol.codec import Codec
        if len(decoded_bytes) >= 6:  
            try:
                frame_len = struct.unpack('>H', decoded_bytes[:2])[0]
                if 0 < frame_len <= len(decoded_bytes) - 2:
                    interleaved = decoded_bytes[2 : 2 + frame_len]

                    fec_encoded = Codec.deinterleave(interleaved, depth=16)

                    if len(fec_encoded) > 0:
                        data_with_crc = Codec.fec_decode(fec_encoded, nsym=30)
                        data, is_valid = self.crc.verify_and_strip(data_with_crc)
                        meta['crc_valid'] = is_valid
                        return data, meta
            except Exception as e:

                print(f"FEC/Decoding error: {e}")
                pass

        data, is_valid = self.crc.verify_and_strip(decoded_bytes)
        meta['crc_valid'] = is_valid
        return data, meta

    def _ofdm_demodulate(self, time_signal: np.ndarray) -> list[np.ndarray]:
        sym_len = self.n_fft + self.cp_length
        num_symbols = len(time_signal) // sym_len

        freq_blocks = []
        for i in range(num_symbols):
            sym = time_signal[i * sym_len : (i + 1) * sym_len]
            sym_no_cp = self._remove_cyclic_prefix(sym)
            freq_block = np.fft.fft(sym_no_cp, n=self.n_fft)
            freq_blocks.append(freq_block)

        return freq_blocks

    def _remove_cyclic_prefix(self, symbol: np.ndarray) -> np.ndarray:
        return symbol[self.cp_length:]

    def _extract_pilots_and_data(self, subcarriers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rx_pilots = subcarriers[self.pilot_carriers]
        rx_data = subcarriers[self.data_carriers]
        return rx_pilots, rx_data, self.pilot_carriers, self.data_carriers

    def get_tx_signal(self) -> np.ndarray | None:
        return self._last_tx_signal

    def get_rx_constellation(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._last_rx_constellation_pre is not None and self._last_rx_constellation_post is not None:
            return self._last_rx_constellation_pre, self._last_rx_constellation_post
        return None

    def get_channel_estimate(self) -> tuple[np.ndarray, np.ndarray] | None:
        return self.channel_est.get_channel_response()

    def transmit(self, data: bytes) -> tuple[np.ndarray, dict]:

        tx_sig = self.encode(data)
        return tx_sig, {'num_samples': len(tx_sig), 'sample_rate': self.sample_rate}

    def receive(self, rx_signal: np.ndarray) -> tuple[bytes, dict]:

        return self.decode(rx_signal)

