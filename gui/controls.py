from __future__ import annotations
import os
import time
import base64
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import TYPE_CHECKING
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal

from config import AirBuddsConfig
from core.morse_transceiver import MorseTransceiver
from core.ofdm_transceiver import OFDMTransceiver
from audio.audio_interface import AudioInterface

if TYPE_CHECKING:
    from gui.dashboard import Dashboard

class ControlPanel:
    def __init__(self, parent_frame: tk.Frame, dashboard: 'Dashboard'):
        self.parent = parent_frame
        self.dashboard = dashboard
        self.config = dashboard.config
        
        self._canvas = tk.Canvas(self.parent, bg=dashboard.accent_color, highlightthickness=0)
        self._scrollbar = ttk.Scrollbar(self.parent, orient=tk.VERTICAL, command=self._canvas.yview)
        self.container = tk.Frame(self._canvas, bg=dashboard.accent_color)
        
        self.container.bind('<Configure>', lambda e: self._canvas.configure(scrollregion=self._canvas.bbox('all')))
        self._canvas.create_window((0, 0), window=self.container, anchor='nw')
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        
        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        def _on_mousewheel(event):
            self._canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        self._canvas.bind_all('<MouseWheel>', _on_mousewheel)
        
        self._listening = False
        self._listening_start_time = 0
        self.audio_interface = AudioInterface(self.dashboard.config.audio)
        
        self._setup_device_selector()
        self._setup_morse_settings()
        self._setup_text_input()
        self._setup_file_input()
        self._setup_rx_display_panel()
        self._setup_wav_file_transfer()
        self._setup_rx_controls()

    def _setup_device_selector(self):
        frame = ttk.LabelFrame(self.container, text="Audio Devices (Hardware)")
        frame.pack(fill=tk.X, padx=5, pady=4)
        
        in_devices = self.audio_interface.get_input_devices()
        out_devices = self.audio_interface.get_output_devices()
        
        tk.Label(frame, text="Microphone:", bg=self.dashboard.accent_color, fg='white').pack(anchor='w', padx=5, pady=(2, 0))
        self.in_dev_map = {name: dev_id for dev_id, name in in_devices}
        in_names = list(self.in_dev_map.keys())
        default_in = in_names[0] if in_names else "Default"
        
        self.in_dev_var = tk.StringVar(value=default_in)
        combo_in = ttk.Combobox(frame, textvariable=self.in_dev_var, values=in_names, state='readonly')
        combo_in.pack(fill=tk.X, padx=5, pady=2)
        combo_in.bind("<<ComboboxSelected>>", self._on_device_changed)
        
        tk.Label(frame, text="Speaker:", bg=self.dashboard.accent_color, fg='white').pack(anchor='w', padx=5, pady=(2, 0))
        self.out_dev_map = {name: dev_id for dev_id, name in out_devices}
        out_names = list(self.out_dev_map.keys())
        default_out = out_names[0] if out_names else "Default"
        
        self.out_dev_var = tk.StringVar(value=default_out)
        combo_out = ttk.Combobox(frame, textvariable=self.out_dev_var, values=out_names, state='readonly')
        combo_out.pack(fill=tk.X, padx=5, pady=2)
        combo_out.bind("<<ComboboxSelected>>", self._on_device_changed)
        
        btn_test = ttk.Button(frame, text="Test Mic Level", command=self._on_test_mic)
        btn_test.pack(pady=4)
        self.mic_level_lbl = tk.Label(frame, text="Mic: Ready", bg=self.dashboard.accent_color, fg='#34d399')
        self.mic_level_lbl.pack()

    def _setup_morse_settings(self):
        self.settings_frame = ttk.LabelFrame(self.container, text="Transceiver Settings")
        self.settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(self.settings_frame, text="Mode:", bg=self.dashboard.accent_color, fg='white').pack(anchor='w', padx=5)
        self.mode_var = tk.StringVar(value=self.config.mode)
        combo_mode = ttk.Combobox(self.settings_frame, textvariable=self.mode_var, values=["morse", "ofdm"], state='readonly')
        combo_mode.pack(fill=tk.X, padx=5, pady=2)
        combo_mode.bind("<<ComboboxSelected>>", self._on_mode_changed)
        
        self.morse_frame = tk.Frame(self.settings_frame, bg=self.dashboard.accent_color)
        
        tk.Label(self.morse_frame, text="Morse Speed (WPM):", bg=self.dashboard.accent_color, fg='white').pack(anchor='w', padx=5)
        self.wpm_var = tk.DoubleVar(value=self.config.morse_wpm)
        scale = tk.Scale(self.morse_frame, from_=5, to=100, orient=tk.HORIZONTAL, variable=self.wpm_var, 
                         bg=self.dashboard.accent_color, fg='white', highlightthickness=0)
        scale.pack(fill=tk.X, padx=5)
        
        lbl = tk.Label(self.morse_frame, text="20-40 WPM for human listening; 100 WPM for fast file transfer.", 
                       bg=self.dashboard.accent_color, fg='#94a3b8', font=('Segoe UI', 8))
        lbl.pack(anchor='w', padx=5, pady=(0, 2))

        self.ofdm_frame = tk.Frame(self.settings_frame, bg=self.dashboard.accent_color)
        
        tk.Label(self.ofdm_frame, text="QAM Order:", bg=self.dashboard.accent_color, fg='white').pack(anchor='w', padx=5)
        self.qam_var = tk.StringVar(value=str(self.config.qam.order))
        combo_qam = ttk.Combobox(self.ofdm_frame, textvariable=self.qam_var, values=["4", "16", "64"], state='readonly')
        combo_qam.pack(fill=tk.X, padx=5, pady=2)
        
        self._on_mode_changed()

    def _on_mode_changed(self, event=None):
        mode = self.mode_var.get()
        self.dashboard.update_mode(mode)
        if mode == "morse":
            self.ofdm_frame.pack_forget()
            self.morse_frame.pack(fill=tk.X, pady=2)
        else:
            self.morse_frame.pack_forget()
            self.ofdm_frame.pack(fill=tk.X, pady=2)

    def _setup_text_input(self):
        frame = ttk.LabelFrame(self.container, text="Text Transmission")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(frame, text="TX Message:", bg=self.dashboard.accent_color, fg='white').pack(anchor='w', padx=5)
        self.text_entry = tk.Entry(frame, bg=self.dashboard.bg_color, fg='white', insertbackground='white')
        self.text_entry.insert(0, "SOS AIRBUDDS")
        self.text_entry.pack(fill=tk.X, padx=5, pady=2)
        
        btn_box = tk.Frame(frame, bg=self.dashboard.accent_color)
        btn_box.pack(fill=tk.X, padx=5, pady=3)
        ttk.Button(btn_box, text="🔊 Transmit (Speaker)", command=self._on_transmit_text).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(btn_box, text="💾 Export (.wav)", command=self._on_export_text_audio).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)

    def _setup_file_input(self):
        frame = ttk.LabelFrame(self.container, text="File Transmission")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.file_path_var = tk.StringVar(value="No file selected")
        tk.Label(frame, textvariable=self.file_path_var, bg=self.dashboard.accent_color, fg='#34d399', anchor='w').pack(fill=tk.X, padx=5)
        
        btn_browse = ttk.Button(frame, text="📂 Select File...", command=self._on_browse_file)
        btn_browse.pack(fill=tk.X, padx=5, pady=2)
        
        btn_box = tk.Frame(frame, bg=self.dashboard.accent_color)
        btn_box.pack(fill=tk.X, padx=5, pady=3)
        ttk.Button(btn_box, text="🔊 Transmit File", command=self._on_transmit_file).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(btn_box, text="💾 Export File (.wav)", command=self._on_export_file_audio).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)

    def _setup_rx_display_panel(self):
        frame = ttk.LabelFrame(self.container, text="Decoded RX Result")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.rx_display = tk.Entry(frame, bg=self.dashboard.bg_color, fg='#34d399', insertbackground='white')
        self.rx_display.pack(fill=tk.X, padx=5, pady=(2, 5))
        
        ttk.Button(frame, text="💾 Save Decoded RX to File", command=self._on_save_rx_to_file).pack(fill=tk.X, padx=5, pady=3)

    def _setup_wav_file_transfer(self):
        frame = ttk.LabelFrame(self.container, text="Audio File Transfer (WAV)")
        frame.pack(fill=tk.X, padx=5, pady=4)
        
        ttk.Button(frame, text="📂 Upload & Decode Audio (.wav)", command=self._on_upload_decode_audio).pack(fill=tk.X, padx=5, pady=3)

    def _setup_rx_controls(self):
        frame = ttk.LabelFrame(self.container, text="Live Microphone Receiver")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(frame, text="Start Listening", command=self._on_start_listening).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=5)
        ttk.Button(frame, text="Stop & Decode", command=self._on_stop_listening).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2, pady=5)

    def _on_device_changed(self, event=None):
        in_name = self.in_dev_var.get()
        out_name = self.out_dev_var.get()
        if in_name in self.in_dev_map:
            self.dashboard.config.audio.device_in = self.in_dev_map[in_name]
        if out_name in self.out_dev_map:
            self.dashboard.config.audio.device_out = self.out_dev_map[out_name]

    def _on_test_mic(self):
        self.mic_level_lbl.config(text="Testing...", fg='#fbbf24')
        def _test():
            try:
                peak, _ = self.audio_interface.test_mic_level(0.4)
                pct = int(min(peak * 100, 100))
                color = '#34d399' if peak > 0.003 else '#f87171'
                self.dashboard.root.after(0, lambda: self.mic_level_lbl.config(text=f"Mic Level: {pct}%", fg=color))
            except Exception as e:
                self.dashboard.root.after(0, lambda: self.mic_level_lbl.config(text="Error", fg='#f87171'))
        threading.Thread(target=_test, daemon=True).start()

    def _on_browse_file(self):
        filepath = filedialog.askopenfilename()
        if filepath:
            self.file_path_var.set(filepath)

    def _apply_transceiver_config(self):
        self.dashboard.config.mode = self.mode_var.get()
        self.dashboard.config.morse_wpm = self.wpm_var.get()
        self.dashboard.config.qam.order = int(self.qam_var.get())

    def _get_transceiver(self):
        if self.dashboard.config.mode == "ofdm":
            return OFDMTransceiver(self.dashboard.config)
        return MorseTransceiver(self.dashboard.config)

    def _read_file_payload(self, filepath: str) -> bytes:
        with open(filepath, 'rb') as f:
            raw = f.read()
            
        # Morse code is case-insensitive, so we MUST use Base32 for files
        # to ensure perfect 1:1 reconstruction of the binary data at the receiver.
        if self.dashboard.config.mode == "morse":
            b32 = "B32:" + base64.b32encode(raw).decode('ascii')
            return b32.encode('utf-8')
            
        try:
            return raw.decode('utf-8').encode('utf-8')
        except Exception:
            b64 = "B64:" + base64.b64encode(raw).decode('ascii')
            return b64.encode('utf-8')

    def _on_transmit_text(self):
        text = self.text_entry.get().strip()
        if not text: return
        self._apply_transceiver_config()
        threading.Thread(target=self._tx_worker, args=(text.encode('utf-8'),), daemon=True).start()

    def _on_transmit_file(self):
        filepath = self.file_path_var.get()
        if not os.path.exists(filepath):
            messagebox.showwarning("Warning", "Please select a valid file first.")
            return
        self._apply_transceiver_config()
        payload = self._read_file_payload(filepath)
        threading.Thread(target=self._tx_worker, args=(payload,), daemon=True).start()

    def _on_export_text_audio(self):
        text = self.text_entry.get().strip()
        if not text: return
        save_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV Audio Files", "*.wav")])
        if not save_path: return
        self._apply_transceiver_config()
        
        transceiver = self._get_transceiver()
        tx_signal = transceiver.encode(text.encode('utf-8'))
        fs = self.dashboard.config.audio.sample_rate
        
        wavfile.write(save_path, fs, (tx_signal * 32767).astype(np.int16))
        self.dashboard.set_status(f"Exported text audio to {os.path.basename(save_path)} ✓")

    def _on_export_file_audio(self):
        filepath = self.file_path_var.get()
        if not os.path.exists(filepath):
            messagebox.showwarning("Warning", "Please select a valid file first.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV Audio Files", "*.wav")])
        if not save_path: return
        
        self._apply_transceiver_config()
        payload = self._read_file_payload(filepath)
        
        transceiver = self._get_transceiver()
        tx_signal = transceiver.encode(payload)
        fs = self.dashboard.config.audio.sample_rate
        
        wavfile.write(save_path, fs, (tx_signal * 32767).astype(np.int16))
        self.dashboard.set_status(f"Exported file audio to {os.path.basename(save_path)} ✓")

    def _tx_worker(self, data: bytes):
        try:
            transceiver = self._get_transceiver()
            tx_signal = transceiver.encode(data)
            dur = len(tx_signal) / self.dashboard.config.audio.sample_rate
            
            self.dashboard.set_status(f"Playing Morse Code ({dur:.1f}s)...")
            self.dashboard.root.after(0, lambda: self.dashboard.update_plots(tx_signal=tx_signal, rx_signal=None))
            
            self.audio_interface.play(tx_signal, blocking=True)
            self.dashboard.set_status(f"Transmission complete! ({dur:.1f}s)")
        except Exception as e:
            self.dashboard.set_status(f"TX Error: {e}")

    def _on_start_listening(self):
        if self._listening: return
        self._listening = True
        self._apply_transceiver_config()
        
        try:
            self.audio_interface.start_stream()
            self._listening_start_time = time.time()
            self._update_listen_timer()
        except Exception as e:
            self._listening = False
            messagebox.showerror("Audio Error", str(e))

    def _update_listen_timer(self):
        if not self._listening: return
        elapsed = int(time.time() - self._listening_start_time)
        self.dashboard.set_status(f"🎤 Listening infinitely... ({elapsed}s elapsed) Click 'Stop' to decode.")
        self.dashboard.root.after(1000, self._update_listen_timer)

    def _on_stop_listening(self):
        if not self._listening: return
        self._listening = False
        self.dashboard.set_status("Processing recorded Morse code...")
        threading.Thread(target=self._rx_decode_worker, daemon=True).start()

    def _handle_auto_save(self, text: str) -> str:
        if text.startswith("B32:") or text.startswith("B64:"):
            save_dir = os.path.join(os.getcwd(), "received_files")
            os.makedirs(save_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(save_dir, f"rx_file_{stamp}.bin")
            try:
                if text.startswith("B64:"):
                    raw_bytes = base64.b64decode(text[4:])
                else:
                    b32_data = text[4:].strip()
                    pad = (8 - (len(b32_data) % 8)) % 8
                    b32_data += "=" * pad
                    raw_bytes = base64.b32decode(b32_data)
                with open(out_path, 'wb') as f:
                    f.write(raw_bytes)
                return f"[Auto-saved file to {out_path}]"
            except Exception as e:
                return f"[Auto-save Failed: {e}] {text}"
        return text

    def _rx_decode_worker(self):
        try:
            rx_signal = self.audio_interface.stop_stream()
            if len(rx_signal) == 0:
                self.dashboard.set_status("No audio recorded.")
                return
                
            transceiver = self._get_transceiver()
            payload, meta = transceiver.decode(rx_signal)
            decoded_str = payload.decode('utf-8', errors='replace')
            
            def _update():
                self.rx_display.delete(0, tk.END)
                display_text = self._handle_auto_save(decoded_str)
                self.rx_display.insert(0, display_text)
                err = meta.get('error', '')
                if err:
                    self.dashboard.set_status(f"Decode Error: {err}")
                elif not payload:
                    self.dashboard.set_status(f"Decode Error: No valid data found (CRC: {meta.get('crc_valid')})")
                else:
                    self.dashboard.set_status(f"Decoded: '{decoded_str}'")
                    
                constellation = None
                if hasattr(transceiver, 'get_rx_constellation'):
                    constellation = transceiver.get_rx_constellation()
                self.dashboard.update_plots(tx_signal=None, rx_signal=rx_signal, constellation=constellation)
                
            self.dashboard.root.after(0, _update)
        except Exception as e:
            self.dashboard.set_status(f"RX Error: {e}")

    def _on_save_rx_to_file(self):
        content = self.rx_display.get().strip()
        if not content:
            messagebox.showwarning("Warning", "No decoded RX content to save.")
            return
            
        save_path = filedialog.asksaveasfilename(
            title="Save Decoded RX Content to File",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if not save_path: return
        
        try:
            if content.startswith("B64:"):
                raw_bytes = base64.b64decode(content[4:])
                with open(save_path, 'wb') as f:
                    f.write(raw_bytes)
            elif content.startswith("B32:"):
                b32_data = content[4:].strip()
                # Base32 requires padding to be a multiple of 8
                pad_len = (8 - (len(b32_data) % 8)) % 8
                b32_data += "=" * pad_len
                raw_bytes = base64.b32decode(b32_data)
                with open(save_path, 'wb') as f:
                    f.write(raw_bytes)
            else:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            messagebox.showinfo("Success", f"File saved successfully to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file: {e}")

    def _on_upload_decode_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV Audio Files", "*.wav"), ("All Files", "*.*")])
        if not file_path: return
        self._apply_transceiver_config()
        
        self.dashboard.set_status("Decoding WAV...")
        threading.Thread(target=self._decode_file_worker, args=(file_path,), daemon=True).start()

    def _decode_file_worker(self, file_path):
        try:
            sr, raw = wavfile.read(file_path)
            audio = raw.astype(np.float32) / 32768.0 if raw.dtype == np.int16 else raw.astype(np.float32)
            if audio.ndim > 1: audio = np.mean(audio, axis=-1)
            audio = audio.flatten()
            
            if sr != self.dashboard.config.audio.sample_rate:
                gcd = math.gcd(sr, self.dashboard.config.audio.sample_rate)
                audio = signal.resample_poly(audio, self.dashboard.config.audio.sample_rate // gcd, sr // gcd).astype(np.float32)
                
            transceiver = self._get_transceiver()
            payload, meta = transceiver.decode(audio)
            
            def _update():
                self.rx_display.delete(0, tk.END)
                text = payload.decode('utf-8', errors='replace')
                display_text = self._handle_auto_save(text)
                self.rx_display.insert(0, display_text)
                if not payload:
                    self.dashboard.set_status(f"WAV Decode Error: No valid data found (CRC: {meta.get('crc_valid')})")
                else:
                    self.dashboard.set_status(f"WAV Decoded: '{text}'")
                    
                constellation = None
                if hasattr(transceiver, 'get_rx_constellation'):
                    constellation = transceiver.get_rx_constellation()
                self.dashboard.update_plots(tx_signal=None, rx_signal=audio, constellation=constellation)
                
            self.dashboard.root.after(0, _update)
        except Exception as e:
            self.dashboard.set_status(f"WAV Decode Error: {e}")
