from __future__ import annotations
import os
import math
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import TYPE_CHECKING, Dict, Any
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal

from config import AirBuddsConfig
from core.ofdm_transceiver import OFDMTransceiver
from core.crc import CRCEngine
from audio.audio_interface import AudioInterface
from dsp.spectral_shift import SpectralShifter
from dsp.watermark import AudioWatermark
from protocol.codec import Codec

if TYPE_CHECKING:
    from gui.dashboard import Dashboard

class ControlPanel:
    """
    Control Panel widget for configuring and operating the AirBudds Transceiver.
    """
    def __init__(self, parent_frame: tk.Frame, dashboard: 'Dashboard'):
        """
        Initialize the control panel.
        
        Parameters
        ----------
        parent_frame : tk.Frame
            The Tkinter frame to embed controls into.
        dashboard : Dashboard
            Reference to the main dashboard instance.
        """
        self.parent = parent_frame
        self.dashboard = dashboard
        self.config = dashboard.config
        
        # Scrollable container so all controls are accessible
        self._canvas = tk.Canvas(self.parent, bg=dashboard.accent_color, highlightthickness=0)
        self._scrollbar = ttk.Scrollbar(self.parent, orient=tk.VERTICAL, command=self._canvas.yview)
        self.container = tk.Frame(self._canvas, bg=dashboard.accent_color)
        
        self.container.bind('<Configure>', lambda e: self._canvas.configure(scrollregion=self._canvas.bbox('all')))
        self._canvas.create_window((0, 0), window=self.container, anchor='nw')
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        
        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Mousewheel scrolling
        def _on_mousewheel(event):
            self._canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        self._canvas.bind_all('<MouseWheel>', _on_mousewheel)
        
        self.variables = {}
        self._listening = False
        
        self._setup_text_input()
        self._setup_wav_file_transfer()
        self._setup_rx_controls()
        self._setup_file_input()
        self._setup_device_selector()
        self._setup_modulation_selector()
        self._setup_mode_toggles()

    def _setup_device_selector(self):
        """Setup audio hardware device selectors for microphone and speaker."""
        frame = ttk.LabelFrame(self.container, text="Audio Devices (Hardware)")
        frame.pack(fill=tk.X, padx=5, pady=4)
        
        audio = AudioInterface(self.dashboard.config.audio)
        in_devices = audio.get_input_devices()
        out_devices = audio.get_output_devices()
        
        # Microphone selector
        tk.Label(frame, text="Microphone (Input):", bg=self.dashboard.accent_color, fg='white').pack(anchor='w', padx=5, pady=(2, 0))
        self.in_dev_map = {name: dev_id for dev_id, name in in_devices}
        in_names = list(self.in_dev_map.keys())
        default_in_name = next((name for name, d_id in self.in_dev_map.items() if d_id == audio.device_in), in_names[0] if in_names else "No input device")
        
        self.in_dev_var = tk.StringVar(value=default_in_name)
        combo_in = ttk.Combobox(frame, textvariable=self.in_dev_var, values=in_names, state='readonly')
        combo_in.pack(fill=tk.X, padx=5, pady=2)
        combo_in.bind("<<ComboboxSelected>>", self._on_device_changed)
        
        # Speaker selector
        tk.Label(frame, text="Speaker (Output):", bg=self.dashboard.accent_color, fg='white').pack(anchor='w', padx=5, pady=(2, 0))
        self.out_dev_map = {name: dev_id for dev_id, name in out_devices}
        out_names = list(self.out_dev_map.keys())
        default_out_name = next((name for name, d_id in self.out_dev_map.items() if d_id == audio.device_out), out_names[0] if out_names else "No output device")
        
        self.out_dev_var = tk.StringVar(value=default_out_name)
        combo_out = ttk.Combobox(frame, textvariable=self.out_dev_var, values=out_names, state='readonly')
        combo_out.pack(fill=tk.X, padx=5, pady=2)
        combo_out.bind("<<ComboboxSelected>>", self._on_device_changed)
        
        # Mic Test Bar
        test_frame = tk.Frame(frame, bg=self.dashboard.accent_color)
        test_frame.pack(fill=tk.X, padx=5, pady=4)
        btn_test = ttk.Button(test_frame, text="Test Mic Level", command=self._on_test_mic)
        btn_test.pack(side=tk.LEFT, padx=2)
        self.mic_level_lbl = tk.Label(test_frame, text="Mic: Ready", bg=self.dashboard.accent_color, fg='#4ec9b0', font=('Segoe UI', 8, 'bold'))
        self.mic_level_lbl.pack(side=tk.LEFT, padx=5)

    def _on_device_changed(self, event=None):
        in_name = self.in_dev_var.get()
        out_name = self.out_dev_var.get()
        if in_name in self.in_dev_map:
            self.dashboard.config.audio.device_in = self.in_dev_map[in_name]
        if out_name in self.out_dev_map:
            self.dashboard.config.audio.device_out = self.out_dev_map[out_name]

    def _on_test_mic(self):
        self.mic_level_lbl.config(text="Testing...", fg='#ffcc00')
        def _test():
            try:
                audio = AudioInterface(self.dashboard.config.audio)
                peak, rms = audio.test_mic_level(0.4)
                pct = int(min(peak * 100, 100))
                if peak > 0.003:
                    status = f"Mic Level: {pct}% (Active 🟢)"
                    color = '#4ec9b0'
                else:
                    status = f"Mic Level: {pct}% (Low/Silent ⚠️ Check Privacy)"
                    color = '#ff6b6b'
                self.dashboard.root.after(0, lambda: self.mic_level_lbl.config(text=status, fg=color))
            except Exception as e:
                self.dashboard.root.after(0, lambda: self.mic_level_lbl.config(text=f"Error: {e}", fg='#ff6b6b'))
        threading.Thread(target=_test, daemon=True).start()

    def _setup_text_input(self):
        """Setup text input field, received display, and transmit button."""
        frame = ttk.LabelFrame(self.container, text="Text Transmission")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(frame, text="TX Message:", bg=self.dashboard.accent_color, fg='white').pack(anchor='w', padx=5, pady=(2, 0))
        self.text_entry = tk.Entry(frame, bg='#1a1a2e', fg='white', insertbackground='white')
        self.text_entry.insert(0, "Hello AirBudds! 🔊")
        self.text_entry.pack(fill=tk.X, padx=5, pady=2)
        
        btn = ttk.Button(frame, text="Transmit Text (Speaker)", command=self._on_transmit_text)
        btn.pack(fill=tk.X, padx=5, pady=3)
        
        tk.Label(frame, text="Decoded RX Result:", bg=self.dashboard.accent_color, fg=self.dashboard.text_color).pack(anchor='w', padx=5, pady=(4, 0))
        self.rx_display = tk.Entry(frame, bg='#1a1a2e', fg='#00ffcc', insertbackground='white')
        self.rx_display.pack(fill=tk.X, padx=5, pady=(2, 5))

    def _setup_wav_file_transfer(self):
        """Setup controls for exporting transmission to WAV and uploading WAV for decoding."""
        frame = ttk.LabelFrame(self.container, text="Audio File Transfer (WAV)")
        frame.pack(fill=tk.X, padx=5, pady=4)
        
        btn_box = tk.Frame(frame, bg=self.dashboard.accent_color)
        btn_box.pack(fill=tk.X, padx=5, pady=3)
        
        btn_export = ttk.Button(btn_box, text="💾 Export Audio (.wav)", command=self._on_export_audio)
        btn_export.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        btn_upload = ttk.Button(btn_box, text="📂 Upload & Decode (.wav)", command=self._on_upload_decode_audio)
        btn_upload.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)
        
        lbl_hint = tk.Label(frame, text="Download audio to play on phone / upload recorded audio to decode.",
                            bg=self.dashboard.accent_color, fg='#8888aa', font=('Segoe UI', 8))
        lbl_hint.pack(anchor='w', padx=5, pady=(0, 2))

    def _setup_file_input(self):
        """Setup file browsing and transmit buttons."""
        frame = ttk.LabelFrame(self.container, text="File Transmission")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.file_path_var = tk.StringVar(value="No file selected")
        lbl = tk.Label(frame, textvariable=self.file_path_var, bg=self.dashboard.accent_color, fg='white', anchor='w')
        lbl.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        btn_box = tk.Frame(frame, bg=self.dashboard.accent_color)
        btn_box.pack(fill=tk.X, padx=5, pady=2)
        btn_browse = ttk.Button(btn_box, text="Browse...", command=self._on_browse_file)
        btn_browse.pack(side=tk.LEFT, padx=2)
        btn_tx = ttk.Button(btn_box, text="Transmit File", command=self._on_transmit_file)
        btn_tx.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)

    def _setup_modulation_selector(self):
        """Setup modulation scheme dropdown."""
        frame = ttk.LabelFrame(self.container, text="Modulation")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.variables['modulation'] = tk.StringVar(value='4-QAM')
        combo = ttk.Combobox(frame, textvariable=self.variables['modulation'], 
                             values=['4-QAM', '16-QAM', '64-QAM'], state='readonly')
        combo.pack(fill=tk.X, padx=5, pady=5)

    def _setup_mode_toggles(self):
        """Setup boolean mode toggles."""
        frame = ttk.LabelFrame(self.container, text="Acoustic Transmission Settings")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Audible Acoustic Mode Indicator
        lbl = tk.Label(frame, text="🔊 Mode: Pure Audible Sound (800 Hz - 3.5 kHz)", 
                       bg='#1e3a1e', fg='#4ec9b0', font=('Segoe UI', 9, 'bold'), padx=5, pady=3)
        lbl.pack(fill=tk.X, padx=5, pady=3)
        
        self.variables['ultrasonic'] = tk.BooleanVar(value=False)
        
        self.variables['longer_audio'] = tk.BooleanVar(value=True)
        cb_long = tk.Checkbutton(frame, text="Longer Audio for High Accuracy (2x repeats)", 
                                 variable=self.variables['longer_audio'], 
                                 bg=self.dashboard.accent_color, fg='white', selectcolor=self.dashboard.bg_color)
        cb_long.pack(anchor='w', padx=5, pady=2)
        
        self.variables['noise_canc'] = tk.BooleanVar(value=False)
        cb2 = tk.Checkbutton(frame, text="LMS Noise Cancellation", variable=self.variables['noise_canc'], 
                             bg=self.dashboard.accent_color, fg='white', selectcolor=self.dashboard.bg_color)
        cb2.pack(anchor='w', padx=5, pady=2)
        
        self.variables['watermark'] = tk.BooleanVar(value=False)
        cb3 = tk.Checkbutton(frame, text="Spread-Spectrum Watermark", variable=self.variables['watermark'], 
                             bg=self.dashboard.accent_color, fg='white', selectcolor=self.dashboard.bg_color)
        cb3.pack(anchor='w', padx=5, pady=2)

    def _setup_rx_controls(self):
        """Setup receiver control buttons."""
        frame = ttk.LabelFrame(self.container, text="Live Microphone Receiver")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        btn_start = ttk.Button(frame, text="Start Listening", command=self._on_start_listening)
        btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=5)
        
        btn_stop = ttk.Button(frame, text="Stop", command=self._on_stop_listening)
        btn_stop.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2, pady=5)



    def _on_browse_file(self):
        filepath = filedialog.askopenfilename()
        if filepath:
            self.file_path_var.set(filepath)

    def _on_transmit_text(self):
        text = self.text_entry.get().strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter text to transmit.")
            return
        threading.Thread(target=self._tx_worker, args=(text.encode('utf-8'),), daemon=True).start()

    def _on_transmit_file(self):
        filepath = self.file_path_var.get()
        if not filepath or filepath == "No file selected":
            messagebox.showwarning("Warning", "Please select a file first.")
            return
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            threading.Thread(target=self._tx_worker, args=(data,), daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file: {e}")

    def _on_export_audio(self):
        """Export current text message to a WAV audio file."""
        text = self.text_entry.get().strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter text to export as audio.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV Audio Files", "*.wav"), ("All Files", "*.*")],
            title="Save Transmitted Acoustic Signal as WAV"
        )
        if not save_path:
            return

        def _export_worker():
            try:
                self.dashboard.set_status(f"Encoding '{text}' to WAV...")
                overrides = self.get_config_overrides()
                order = int(overrides['modulation'].split('-')[0])
                self.dashboard.config.qam.order = order
                self.dashboard.config.ofdm.cp_length = overrides['cp_length']
                self.dashboard.config.ultrasonic.enabled = overrides['ultrasonic']
                self.dashboard.config.watermark.enabled = overrides['watermark']

                longer = self.variables.get('longer_audio', tk.BooleanVar(value=True)).get()
                self.dashboard.config.ofdm.symbol_repeats = 2 if longer else 1

                transceiver = OFDMTransceiver(self.dashboard.config)
                tx_signal = transceiver.encode(text.encode('utf-8'))

                if overrides['watermark']:
                    wm = AudioWatermark(self.dashboard.config.watermark, self.dashboard.config.audio.sample_rate)
                    tx_signal = wm.embed(tx_signal)

                fs = self.dashboard.config.audio.sample_rate
                lead_in = np.zeros(int(0.25 * fs), dtype=np.float32)
                lead_out = np.zeros(int(0.25 * fs), dtype=np.float32)
                full_signal = np.concatenate([lead_in, tx_signal, lead_out])

                max_v = np.max(np.abs(full_signal))
                norm_sig = (full_signal / max_v * 0.90) if max_v > 0 else full_signal
                pcm_data = (norm_sig * 32767.0).astype(np.int16)

                wavfile.write(save_path, fs, pcm_data)
                dur = len(pcm_data) / fs

                ideal = transceiver.qam.get_constellation_points()
                self.dashboard.root.after(0, lambda: self.dashboard.update_plots(
                    tx_signal=full_signal, rx_signal=None,
                    constellation=ideal, channel_H=None, ideal_points=ideal
                ))
                self.dashboard.set_status(f"Exported to {os.path.basename(save_path)} ({dur:.2f}s, {order}-QAM) ✓")
                messagebox.showinfo("Export Successful", f"Acoustic signal saved successfully:\n\n{save_path}\nDuration: {dur:.2f} s\nSample Rate: {fs} Hz\n\nYou can now play this file on any phone or speaker!")
            except Exception as e:
                self.dashboard.set_status(f"Export Error: {e}")
                messagebox.showerror("Export Error", f"Could not save WAV file:\n{e}")

        threading.Thread(target=_export_worker, daemon=True).start()

    def _on_upload_decode_audio(self):
        """Upload a WAV file, automatically convert/resample, and decode the transmission."""
        file_path = filedialog.askopenfilename(
            filetypes=[("WAV Audio Files", "*.wav"), ("All Files", "*.*")],
            title="Select Audio File to Decode (WAV)"
        )
        if not file_path:
            return

        def _upload_worker():
            try:
                self.dashboard.set_status(f"Loading {os.path.basename(file_path)}...")
                sr, raw_data = wavfile.read(file_path)

                if raw_data.dtype == np.int16:
                    audio = raw_data.astype(np.float32) / 32768.0
                elif raw_data.dtype == np.int32:
                    audio = raw_data.astype(np.float32) / 2147483648.0
                elif raw_data.dtype == np.uint8:
                    audio = (raw_data.astype(np.float32) - 128.0) / 128.0
                else:
                    audio = raw_data.astype(np.float32)

                if audio.ndim > 1:
                    audio = np.mean(audio, axis=-1)
                audio = audio.flatten()

                target_sr = self.dashboard.config.audio.sample_rate
                if sr != target_sr:
                    self.dashboard.set_status(f"Resampling from {sr} Hz to {target_sr} Hz...")
                    gcd = math.gcd(sr, target_sr)
                    audio = signal.resample_poly(audio, target_sr // gcd, sr // gcd).astype(np.float32)

                overrides = self.get_config_overrides()
                order = int(overrides['modulation'].split('-')[0])
                self.dashboard.config.qam.order = order
                self.dashboard.config.ofdm.cp_length = overrides['cp_length']

                longer = self.variables.get('longer_audio', tk.BooleanVar(value=True)).get()
                self.dashboard.config.ofdm.symbol_repeats = 2 if longer else 1

                transceiver = OFDMTransceiver(self.dashboard.config)
                payload, meta = transceiver.decode(audio)
                is_valid = meta.get('crc_valid', False)

                decoded_str = payload.decode('utf-8', errors='replace')
                rx_const = transceiver.get_rx_constellation()
                post_eq = rx_const[1] if rx_const is not None else None
                ideal = transceiver.qam.get_constellation_points()
                H_resp = transceiver.get_channel_estimate()
                H_mag = H_resp[0] if H_resp is not None else None

                def _update_ui():
                    self.rx_display.delete(0, tk.END)
                    self.rx_display.insert(0, decoded_str)
                    self.dashboard.set_status(f"File Decoded | CRC: {'✓ VALID' if is_valid else '✗ FAIL'} | Result: '{decoded_str}'")
                    self.dashboard.update_plots(
                        tx_signal=None,
                        rx_signal=audio,
                        constellation=post_eq if post_eq is not None else ideal,
                        channel_H=H_mag,
                        ideal_points=ideal
                    )
                    if is_valid:
                        messagebox.showinfo("Decode Successful", f"CRC Check: ✓ VALID\n\nDecoded Message:\n{decoded_str}")
                    else:
                        messagebox.showwarning("CRC Mismatch", f"CRC Check: ✗ FAILED\n(Signal may have noise or distortion)\n\nRaw extracted string:\n{decoded_str}")

                self.dashboard.root.after(0, _update_ui)
            except Exception as e:
                self.dashboard.set_status(f"Upload Decode Error: {e}")
                messagebox.showerror("Decode Error", f"Failed to decode audio file:\n{e}")

        threading.Thread(target=_upload_worker, daemon=True).start()

    def _tx_worker(self, data: bytes):
        try:
            self.dashboard.set_status("Encoding OFDM signal...")
            overrides = self.get_config_overrides()
            mod_str = overrides['modulation']
            order = int(mod_str.split('-')[0])
            self.dashboard.config.qam.order = order
            self.dashboard.config.ofdm.cp_length = overrides['cp_length']
            self.dashboard.config.ultrasonic.enabled = overrides['ultrasonic']
            self.dashboard.config.watermark.enabled = overrides['watermark']

            longer = self.variables.get('longer_audio', tk.BooleanVar(value=True)).get()
            self.dashboard.config.ofdm.symbol_repeats = 2 if longer else 1

            transceiver = OFDMTransceiver(self.dashboard.config)
            tx_signal = transceiver.encode(data)

            if overrides['watermark']:
                wm = AudioWatermark(self.dashboard.config.watermark, self.dashboard.config.audio.sample_rate)
                tx_signal = wm.embed(tx_signal)

            # Update dashboard plots with TX waveform and ideal constellation
            ideal = transceiver.qam.get_constellation_points()
            self.dashboard.root.after(0, lambda: self.dashboard.update_plots(
                tx_signal=tx_signal, rx_signal=None,
                constellation=ideal, channel_H=None, ideal_points=ideal
            ))

            dur = len(tx_signal) / self.dashboard.config.audio.sample_rate
            self.dashboard.set_status(f"Playing acoustic signal via speaker ({dur:.2f}s, {order}-QAM)...")

            audio = AudioInterface(self.dashboard.config.audio)
            audio.play(tx_signal, blocking=True)
            self.dashboard.set_status(f"Transmission complete! ({len(data)} bytes, {order}-QAM, {dur:.2f}s)")
        except Exception as e:
            self.dashboard.set_status(f"TX Error: {e}")


    def _on_start_listening(self):
        if self._listening:
            return
        self._listening = True
        self.dashboard.set_status("Listening on microphone for acoustic OFDM transmission (6s)...")
        threading.Thread(target=self._rx_worker, daemon=True).start()

    def _on_stop_listening(self):
        self._listening = False
        self.dashboard.set_status("Stopped listening.")

    def _rx_worker(self):
        try:
            audio = AudioInterface(self.dashboard.config.audio)
            rx_signal = audio.record(6.0)
            if not self._listening:
                return
            self.dashboard.set_status("Processing microphone signal...")

            overrides = self.get_config_overrides()
            if overrides['noise_cancellation']:
                lms = LMSFilter(self.dashboard.config.lms)
                rx_signal = lms.cancel_noise(rx_signal)

            longer = self.variables.get('longer_audio', tk.BooleanVar(value=True)).get()
            self.dashboard.config.ofdm.symbol_repeats = 2 if longer else 1

            transceiver = OFDMTransceiver(self.dashboard.config)
            payload, meta = transceiver.decode(rx_signal)
            is_valid = meta.get('crc_valid', False)

            decoded_str = payload.decode('utf-8', errors='replace')
            rx_const = transceiver.get_rx_constellation()
            post_eq = rx_const[1] if rx_const is not None else None
            ideal = transceiver.qam.get_constellation_points()
            H_resp = transceiver.get_channel_estimate()
            H_mag = H_resp[0] if H_resp is not None else None

            def _update_ui():
                self.rx_display.delete(0, tk.END)
                self.rx_display.insert(0, decoded_str)
                self.dashboard.set_status(f"Packet Decoded | CRC: {'✓ VALID' if is_valid else '✗ FAIL'} | Result: '{decoded_str}'")
                self.dashboard.update_plots(
                    tx_signal=None,
                    rx_signal=rx_signal,
                    constellation=post_eq if post_eq is not None else ideal,
                    channel_H=H_mag,
                    ideal_points=ideal
                )
                self._listening = False

            self.dashboard.root.after(0, _update_ui)
        except Exception as e:
            self._listening = False
            self.dashboard.set_status(f"Live RX: {e}")

    def get_config_overrides(self) -> Dict[str, Any]:
        """
        Get current UI parameter values.
        """
        return {
            'modulation': self.variables['modulation'].get(),
            'ultrasonic': self.variables['ultrasonic'].get(),
            'noise_cancellation': self.variables['noise_canc'].get(),
            'watermark': self.variables['watermark'].get(),
            'cp_length': getattr(self.dashboard.config.ofdm, 'cp_length', 256),
            'lms_step_size': 1e-3
        }
