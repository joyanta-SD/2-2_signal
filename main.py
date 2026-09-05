#!/usr/bin/env python3
"""
AirBudds — Acoustic OFDM Data Transceiver & Signal Analyzer
=============================================================
Course  : Signals & Linear Systems
Team    : 8J3D9
Members : Joyanta Sutradhar (2305083), Dipbroto Karmokar Dip (2305089)

Application entry point.  This module wires every subsystem together and
provides three operating modes:

1. **GUI mode** (default) — launches the interactive Tkinter dashboard.
2. **CLI transmit** — encodes a message/file and plays it through the speaker.
3. **CLI receive** — records from the microphone and decodes the data.
4. **RIR simulation** — runs an offline loopback test through a synthetic channel.

Usage
-----
    python main.py                          # launch GUI
    python main.py --tx "Hello AirBudds"    # transmit text over speaker
    python main.py --rx --duration 5        # record 5 s and decode
    python main.py --sim --preset hallway   # offline RIR simulation
    python main.py --ultrasonic --tx "Hi"   # ultrasonic silent mode
"""

from __future__ import annotations

import argparse
import sys
import time
import numpy as np

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from config import AirBuddsConfig, DEFAULT_CONFIG
from core.ofdm_transceiver import OFDMTransceiver
from core.crc import CRCEngine
from protocol.codec import Codec
from protocol.frame_builder import FrameBuilder
from protocol.retry_manager import RetryManager
from dsp.rir_simulator import RIRSimulator
from dsp.lms_filter import LMSFilter
from dsp.spectral_shift import SpectralShifter
from dsp.watermark import AudioWatermark


# ──────────────────────────────────────────────────────────────────────────────
#  CLI Argument Parser
# ──────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    """Construct the command‑line argument parser."""
    p = argparse.ArgumentParser(
        prog="AirBudds",
        description="Acoustic OFDM Data Transceiver & Signal Analyzer",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--gui", action="store_true", default=True,
        help="Launch the interactive dashboard (default).",
    )
    mode.add_argument(
        "--tx", type=str, metavar="MESSAGE",
        help="Transmit a text message over the speaker.",
    )
    mode.add_argument(
        "--tx-file", type=str, metavar="FILEPATH",
        help="Transmit a file over the speaker.",
    )
    mode.add_argument(
        "--rx", action="store_true",
        help="Record from microphone and decode.",
    )
    mode.add_argument(
        "--sim", action="store_true",
        help="Run offline RIR simulation (no audio hardware needed).",
    )

    # Audio file import/export
    p.add_argument(
        "--export-audio", type=str, metavar="OUT_WAV",
        help="Export encoded acoustic signal to a WAV file (use with --tx for custom message).",
    )
    p.add_argument(
        "--decode-audio", type=str, metavar="IN_WAV",
        help="Load an audio file (WAV), decode it, and display the result.",
    )

    # Common options
    p.add_argument(
        "--duration", type=float, default=5.0,
        help="Recording duration in seconds (for --rx mode). Default: 5.0",
    )
    p.add_argument(
        "--qam", type=int, choices=[4, 16, 64], default=4,
        help="QAM modulation order. Default: 4 (QPSK)",
    )
    p.add_argument(
        "--ultrasonic", action="store_true",
        help="Enable ultrasonic (18–20 kHz) silent transfer mode.",
    )
    p.add_argument(
        "--preset", type=str, default="small_room",
        choices=["anechoic", "small_room", "hallway", "open_air"],
        help="RIR simulation preset. Default: small_room",
    )
    p.add_argument(
        "--snr", type=float, default=30.0,
        help="Simulated channel SNR in dB. Default: 30.0",
    )
    p.add_argument(
        "--noise-cancel", action="store_true",
        help="Enable LMS adaptive noise cancellation on receive.",
    )
    p.add_argument(
        "--watermark", action="store_true",
        help="Embed spread‑spectrum watermark in transmission.",
    )
    p.add_argument(
        "--retries", type=int, default=5,
        help="Maximum retransmission attempts. Default: 5",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose diagnostic output.",
    )
    return p


# ──────────────────────────────────────────────────────────────────────────────
#  Mode Handlers
# ──────────────────────────────────────────────────────────────────────────────
def run_gui(config: AirBuddsConfig) -> None:
    """Launch the interactive Tkinter dashboard."""
    from gui.dashboard import Dashboard
    print("[AirBudds] Launching GUI dashboard …")
    dashboard = Dashboard(config)
    dashboard.run()


def run_transmit(message: str, config: AirBuddsConfig, verbose: bool = False) -> None:
    """Encode a text message and play it through the speaker."""
    from audio.audio_interface import AudioInterface

    print(f"[AirBudds TX] Encoding message: {message!r}")
    print(f"[AirBudds TX] QAM order: {config.qam.order}-QAM")
    print(f"[AirBudds TX] Ultrasonic: {'ON' if config.ultrasonic.enabled else 'OFF'}")

    # Build transceiver and encode
    transceiver = OFDMTransceiver(config)
    data = message.encode("utf-8")

    # Encode to acoustic signal (transceiver.encode appends CRC and frames)
    tx_signal = transceiver.encode(data)

    if verbose:
        print(f"[AirBudds TX] Signal length: {len(tx_signal)} samples")
        print(f"[AirBudds TX] Duration: {len(tx_signal)/config.audio.sample_rate:.3f} s")
        print(f"[AirBudds TX] Peak amplitude: {np.max(np.abs(tx_signal)):.4f}")

    # Ultrasonic upshift
    if config.ultrasonic.enabled:
        shifter = SpectralShifter(config.ultrasonic, config.audio.sample_rate)
        tx_signal = shifter.encode_ultrasonic(tx_signal)
        print("[AirBudds TX] Signal shifted to ultrasonic band (18–20 kHz)")

    # Embed watermark
    if config.watermark.enabled:
        wm = AudioWatermark(config.watermark, config.audio.sample_rate)
        tx_signal = wm.embed(tx_signal)
        print("[AirBudds TX] Spread‑spectrum watermark embedded")

    # Play through speaker
    audio = AudioInterface(config.audio)
    print("[AirBudds TX] Playing signal …")
    audio.play(tx_signal, blocking=True)
    print("[AirBudds TX] Transmission complete ✓")


def run_transmit_file(filepath: str, config: AirBuddsConfig, verbose: bool = False) -> None:
    """Read a file, encode it, and play through the speaker."""
    from audio.audio_interface import AudioInterface

    print(f"[AirBudds TX] Encoding file: {filepath}")
    bits, metadata = Codec.file_to_bits(filepath)
    print(f"[AirBudds TX] File size: {metadata['size']} bytes, name: {metadata['filename']}")

    transceiver = OFDMTransceiver(config)
    data = Codec.bits_to_bytes(bits)

    tx_signal = transceiver.encode(data)

    if config.ultrasonic.enabled:
        shifter = SpectralShifter(config.ultrasonic, config.audio.sample_rate)
        tx_signal = shifter.encode_ultrasonic(tx_signal)

    if config.watermark.enabled:
        wm = AudioWatermark(config.watermark, config.audio.sample_rate)
        tx_signal = wm.embed(tx_signal)

    audio = AudioInterface(config.audio)
    print("[AirBudds TX] Playing signal …")
    audio.play(tx_signal, blocking=True)
    print("[AirBudds TX] File transmission complete ✓")


def run_receive(config: AirBuddsConfig, duration: float = 5.0,
                noise_cancel: bool = False, verbose: bool = False) -> None:
    """Record from the microphone, decode, and display the result."""
    from audio.audio_interface import AudioInterface

    audio = AudioInterface(config.audio)
    print(f"[AirBudds RX] Recording for {duration:.1f} seconds …")
    rx_signal = audio.record(duration)
    print(f"[AirBudds RX] Recorded {len(rx_signal)} samples")

    # Watermark detection
    if config.watermark.enabled:
        wm = AudioWatermark(config.watermark, config.audio.sample_rate)
        detected, corr = wm.detect(rx_signal)
        print(f"[AirBudds RX] Watermark: {'DETECTED' if detected else 'not found'} "
              f"(correlation: {corr:.4f})")

    # Ultrasonic downshift
    if config.ultrasonic.enabled:
        shifter = SpectralShifter(config.ultrasonic, config.audio.sample_rate)
        rx_signal = shifter.decode_ultrasonic(rx_signal)
        print("[AirBudds RX] Signal downshifted from ultrasonic band")

    # LMS noise cancellation
    if noise_cancel:
        lms = LMSFilter(config.lms)
        # Use the first portion as noise reference
        noise_samples = int(config.lms.noise_estimation_ms * config.audio.sample_rate / 1000)
        if len(rx_signal) > noise_samples:
            lms.estimate_noise_profile(rx_signal[:noise_samples])
            rx_signal = lms.cancel_noise(rx_signal)
            print("[AirBudds RX] LMS noise cancellation applied")

    # Decode
    transceiver = OFDMTransceiver(config)
    try:
        payload, metadata = transceiver.decode(rx_signal)

        if verbose:
            for key, val in metadata.items():
                print(f"[AirBudds RX]   {key}: {val}")

        crc_valid = metadata.get('crc_valid', False)

        if crc_valid:
            text = payload.decode("utf-8", errors="replace")
            print(f"[AirBudds RX] ✓ CRC valid — Decoded message: {text!r}")
        else:
            print("[AirBudds RX] ✗ CRC check failed — data may be corrupted")
            text = payload.decode("utf-8", errors="replace")
            print(f"[AirBudds RX]   Raw decoded: {text!r}")

    except Exception as exc:
        print(f"[AirBudds RX] Decoding failed: {exc}")
        if verbose:
            import traceback
            traceback.print_exc()


def run_export_audio(message: str, out_path: str, config: AirBuddsConfig) -> None:
    """Encode message and write directly to a WAV file."""
    import scipy.io.wavfile as wavfile
    print(f"[AirBudds Export] Encoding message: {message!r}")
    transceiver = OFDMTransceiver(config)
    tx_signal = transceiver.encode(message.encode("utf-8"))

    fs = config.audio.sample_rate
    lead = np.zeros(int(0.25 * fs), dtype=np.float32)
    full_sig = np.concatenate([lead, tx_signal, lead])
    max_v = np.max(np.abs(full_sig))
    norm_sig = (full_sig / max_v * 0.90) if max_v > 0 else full_sig
    pcm = (norm_sig * 32767.0).astype(np.int16)

    wavfile.write(out_path, fs, pcm)
    dur = len(pcm) / fs
    print(f"[AirBudds Export] ✓ Saved to {out_path} ({dur:.2f} s, {fs} Hz)")


def run_decode_audio(in_path: str, config: AirBuddsConfig, verbose: bool = False) -> None:
    """Read a WAV file, resample/flatten if needed, and decode."""
    import scipy.io.wavfile as wavfile
    import scipy.signal as signal
    import math
    print(f"[AirBudds Decode] Loading audio file: {in_path}")
    sr, raw = wavfile.read(in_path)
    if raw.dtype == np.int16:
        audio = raw.astype(np.float32) / 32768.0
    elif raw.dtype == np.int32:
        audio = raw.astype(np.float32) / 2147483648.0
    elif raw.dtype == np.uint8:
        audio = (raw.astype(np.float32) - 128.0) / 128.0
    else:
        audio = raw.astype(np.float32)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    audio = audio.flatten()

    target_sr = config.audio.sample_rate
    if sr != target_sr:
        print(f"[AirBudds Decode] Resampling from {sr} Hz to {target_sr} Hz...")
        gcd = math.gcd(sr, target_sr)
        audio = signal.resample_poly(audio, target_sr // gcd, sr // gcd).astype(np.float32)

    transceiver = OFDMTransceiver(config)
    payload, meta = transceiver.decode(audio)
    is_valid = meta.get('crc_valid', False)
    text = payload.decode("utf-8", errors="replace")

    if is_valid:
        print(f"[AirBudds Decode] ✓ CRC VALID — Decoded message: {text!r}")
    else:
        print(f"[AirBudds Decode] ✗ CRC FAILED — Extracted data: {text!r}")

    if verbose:
        for k, v in meta.items():
            print(f"  {k}: {v}")


def run_simulation(config: AirBuddsConfig, verbose: bool = False) -> None:
    """Run an offline loopback test through a simulated room channel."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("  AirBudds — Offline RIR Simulation")
    print(f"  Preset : {config.rir.preset}")
    print(f"  SNR    : {config.rir.snr_db} dB")
    print(f"  QAM    : {config.qam.order}-QAM")
    print("=" * 60)

    # Original message
    test_message = "Hello from AirBudds! 🔊 Team 8J3D9"
    print(f"\n[SIM] Original message: {test_message!r}")

    # Encode
    transceiver = OFDMTransceiver(config)
    data = test_message.encode("utf-8")
    tx_signal = transceiver.encode(data)
    print(f"[SIM] TX signal: {len(tx_signal)} samples, "
          f"{len(tx_signal)/config.audio.sample_rate:.3f} s")

    # Simulate channel
    rir = RIRSimulator(config.rir, config.audio.sample_rate)
    h = rir.get_impulse_response()
    rx_signal = rir.simulate(tx_signal)
    print(f"[SIM] Channel applied: {config.rir.preset}, "
          f"h[n] has {len(h)} taps, SNR={config.rir.snr_db} dB")
    print(f"[SIM] RX signal: {len(rx_signal)} samples")

    # Decode
    try:
        decoded_data, metadata = transceiver.decode(rx_signal)
        payload, crc_valid = crc_engine.verify_and_strip(decoded_data)

        if crc_valid:
            decoded_text = payload.decode("utf-8", errors="replace")
            print(f"[SIM] ✓ Decoded: {decoded_text!r}")
            print(f"[SIM] ✓ Match: {decoded_text == test_message}")
        else:
            print("[SIM] ✗ CRC failed")

        if verbose and metadata:
            for k, v in metadata.items():
                print(f"[SIM]   {k}: {v}")

    except Exception as exc:
        print(f"[SIM] ✗ Decode failed: {exc}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("AirBudds — RIR Simulation Results", fontsize=14, fontweight="bold")
    fig.patch.set_facecolor("#1a1a2e")

    for ax in axes.flat:
        ax.set_facecolor("#0f3460")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#16213e")

    # TX waveform
    t_tx = np.arange(len(tx_signal)) / config.audio.sample_rate * 1000
    axes[0, 0].plot(t_tx, tx_signal, color="#00d2ff", linewidth=0.4)
    axes[0, 0].set_title("TX Signal")
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].set_ylabel("Amplitude")

    # RX waveform
    t_rx = np.arange(len(rx_signal)) / config.audio.sample_rate * 1000
    axes[0, 1].plot(t_rx, rx_signal, color="#e94560", linewidth=0.4)
    axes[0, 1].set_title("RX Signal (after channel)")
    axes[0, 1].set_xlabel("Time (ms)")

    # Impulse response
    t_h = np.arange(len(h)) / config.audio.sample_rate * 1000
    axes[1, 0].stem(t_h, h, linefmt="#e94560", markerfmt="o", basefmt="#16213e")
    axes[1, 0].set_title(f"Room Impulse Response — {config.rir.preset}")
    axes[1, 0].set_xlabel("Time (ms)")
    axes[1, 0].set_ylabel("h[n]")

    # Spectrum comparison
    N_fft = 4096
    freq = np.fft.rfftfreq(N_fft, 1 / config.audio.sample_rate) / 1000
    TX_spec = 20 * np.log10(np.abs(np.fft.rfft(tx_signal, N_fft)) + 1e-12)
    RX_spec = 20 * np.log10(np.abs(np.fft.rfft(rx_signal[:len(tx_signal)], N_fft)) + 1e-12)
    axes[1, 1].plot(freq, TX_spec, color="#00d2ff", alpha=0.7, label="TX", linewidth=0.6)
    axes[1, 1].plot(freq, RX_spec, color="#e94560", alpha=0.7, label="RX", linewidth=0.6)
    axes[1, 1].set_title("Spectrum Comparison")
    axes[1, 1].set_xlabel("Frequency (kHz)")
    axes[1, 1].set_ylabel("Magnitude (dB)")
    axes[1, 1].legend(facecolor="#16213e", edgecolor="white", labelcolor="white")

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """Application entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Build configuration from CLI flags
    config = AirBuddsConfig()
    config.qam.order = args.qam
    config.ultrasonic.enabled = args.ultrasonic
    config.rir.preset = args.preset
    config.rir.snr_db = args.snr
    config.watermark.enabled = args.watermark
    config.protocol.max_retries = args.retries

    print("==========================================================")
    print(f"   AirBudds - Acoustic OFDM Transceiver  v{config.version}")
    print(f"   Team {config.team_name}: {', '.join(config.members)}")
    print("==========================================================")

    # Dispatch to the appropriate mode
    if args.export_audio:
        msg = args.tx if args.tx else "Hello AirBudds! 🔊"
        run_export_audio(msg, args.export_audio, config)
    elif args.decode_audio:
        run_decode_audio(args.decode_audio, config, verbose=args.verbose)
    elif args.tx:
        run_transmit(args.tx, config, verbose=args.verbose)
    elif args.tx_file:
        run_transmit_file(args.tx_file, config, verbose=args.verbose)
    elif args.rx:
        run_receive(config, duration=args.duration,
                    noise_cancel=args.noise_cancel, verbose=args.verbose)
    elif args.sim:
        run_simulation(config, verbose=args.verbose)
    else:
        # Default: GUI mode
        run_gui(config)


if __name__ == "__main__":
    main()
