#!/usr/bin/env python3
import argparse
import sys
import numpy as np

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from config import AirBuddsConfig, DEFAULT_CONFIG
from core.morse_transceiver import MorseTransceiver
from core.ofdm_transceiver import OFDMTransceiver

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="AirBudds",
        description="AirBudds Morse Code Transceiver",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--gui", action="store_true", default=True, help="Launch the interactive dashboard (default).")
    mode.add_argument("--tx", type=str, metavar="MESSAGE", help="Transmit a text message over the speaker.")
    mode.add_argument("--rx", action="store_true", help="Record from microphone and decode.")
    
    
    p.add_argument("--mode", type=str, choices=["morse", "ofdm"], default="morse", help="Transceiver mode (morse or ofdm).")
    p.add_argument("--export-audio", type=str, metavar="OUT_WAV", help="Export encoded acoustic signal to a WAV file.")
    p.add_argument("--decode-audio", type=str, metavar="IN_WAV", help="Load an audio file (WAV), decode it.")
    
    p.add_argument("--wpm", type=float, default=20.0, help="Morse code speed in Words Per Minute.")
    p.add_argument("--freq", type=float, default=800.0, help="Morse code tone frequency in Hz.")
    p.add_argument("--duration", type=float, default=10.0, help="Recording duration in seconds (for --rx mode).")
    
    return p

def run_gui(config: AirBuddsConfig) -> None:
    from gui.dashboard import Dashboard
    print("[AirBudds] Launching Morse GUI dashboard...")
    dashboard = Dashboard(config)
    dashboard.run()

def _get_transceiver(config: AirBuddsConfig, mode: str):
    if mode == "ofdm":
        return OFDMTransceiver(config)
    return MorseTransceiver(config)

def run_transmit(message: str, config: AirBuddsConfig, mode: str) -> None:
    from audio.audio_interface import AudioInterface
    print(f"[AirBudds TX] Encoding message: {message!r} (Mode: {mode})")
    
    transceiver = _get_transceiver(config, mode)
    tx_signal = transceiver.encode(message.encode("utf-8"))
    
    audio = AudioInterface(config.audio)
    print("[AirBudds TX] Playing signal ...")
    audio.play(tx_signal, blocking=True)
    print("[AirBudds TX] Transmission complete ✓")

def run_receive(config: AirBuddsConfig, duration: float = 10.0, mode: str = "morse") -> None:
    from audio.audio_interface import AudioInterface
    audio = AudioInterface(config.audio)
    print(f"[AirBudds RX] Recording for {duration:.1f} seconds ... (Mode: {mode})")
    rx_signal = audio.record(duration, blocking=True)
    
    transceiver = _get_transceiver(config, mode)
    payload, meta = transceiver.decode(rx_signal)
    
    err = meta.get('error', '')
    if err:
        print(f"[AirBudds RX] ✗ Decode failed: {err}")
    elif not payload:
        print(f"[AirBudds RX] ✗ Decode failed: No valid data found (CRC failed: {meta.get('crc_valid')})")
    else:
        text = payload.decode("utf-8", errors="replace")
        print(f"[AirBudds RX] ✓ Decoded message: {text!r}")

def run_export_audio(message: str, out_path: str, config: AirBuddsConfig, mode: str) -> None:
    import scipy.io.wavfile as wavfile
    print(f"[AirBudds Export] Encoding message: {message!r}")
    transceiver = _get_transceiver(config, mode)
    tx_signal = transceiver.encode(message.encode("utf-8"))
    
    fs = config.audio.sample_rate
    wavfile.write(out_path, fs, (tx_signal * 32767).astype(np.int16))
    print(f"[AirBudds Export] ✓ Saved to {out_path}")

def run_decode_audio(in_path: str, config: AirBuddsConfig, mode: str) -> None:
    import scipy.io.wavfile as wavfile
    import scipy.signal as signal
    import math
    print(f"[AirBudds Decode] Loading audio file: {in_path}")
    sr, raw = wavfile.read(in_path)
    audio = raw.astype(np.float32) / 32768.0 if raw.dtype == np.int16 else raw.astype(np.float32)
    if audio.ndim > 1: audio = np.mean(audio, axis=-1)
    audio = audio.flatten()
    
    if sr != config.audio.sample_rate:
        gcd = math.gcd(sr, config.audio.sample_rate)
        audio = signal.resample_poly(audio, config.audio.sample_rate // gcd, sr // gcd).astype(np.float32)
        
    transceiver = _get_transceiver(config, mode)
    payload, meta = transceiver.decode(audio)
    
    err = meta.get('error', '')
    if err:
        print(f"[AirBudds Decode] ✗ Decode failed: {err}")
    elif not payload:
        print(f"[AirBudds RX] ✗ Decode failed: No valid data found (CRC valid: {meta.get('crc_valid')})")
    else:
        text = payload.decode("utf-8", errors="replace")
        print(f"[AirBudds Decode] ✓ Decoded message: {text!r}")

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    
    config = AirBuddsConfig()
    config.morse_wpm = args.wpm
    config.morse_freq = args.freq
    
    if args.export_audio:
        msg = args.tx if args.tx else "SOS"
        run_export_audio(msg, args.export_audio, config, args.mode)
    elif args.decode_audio:
        run_decode_audio(args.decode_audio, config, args.mode)
    elif args.tx:
        run_transmit(args.tx, config, args.mode)
    elif args.rx:
        run_receive(config, duration=args.duration, mode=args.mode)
    else:
        run_gui(config)

if __name__ == "__main__":
    main()
