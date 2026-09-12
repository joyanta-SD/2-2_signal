from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from typing import Optional, TYPE_CHECKING, Any

from config import AirBuddsConfig, DEFAULT_CONFIG
from core.morse_transceiver import MorseTransceiver
from gui.waveform_plot import WaveformPlot
from gui.spectrogram_plot import SpectrogramPlot
from gui.constellation_plot import ConstellationPlot
from gui.controls import ControlPanel

class Dashboard:
    def __init__(self, config: Optional[AirBuddsConfig] = None):
        self.config = config if config is not None else DEFAULT_CONFIG
        self.root = tk.Tk()
        
        self.bg_color = '#1a1a2e'
        self.accent_color = '#16213e'
        self.text_color = '#e94560'
        self.plot_bg_color = '#0f3460'
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('.', background=self.bg_color, foreground='white')
        self.style.configure('TLabelframe', background=self.accent_color, foreground=self.text_color)
        self.style.configure('TLabelframe.Label', background=self.accent_color, foreground=self.text_color)
        self.style.configure('TButton', background=self.plot_bg_color, foreground='white')
        
        self.root.configure(bg=self.bg_color)
        
        self._setup_window()
        self._setup_menu()
        self._setup_layout()
        self._setup_status_bar()

    def _setup_window(self):
        self.root.title('AirBudds — Morse Code Transceiver')
        self.root.geometry('1400x900')
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=3)
        self.root.rowconfigure(0, weight=1)

    def _setup_menu(self):
        menubar = tk.Menu(self.root)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About AirBudds", "AirBudds Morse Transceiver"))
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)

    def _setup_layout(self):
        self.left_panel = tk.Frame(self.root, bg=self.accent_color, width=350)
        self.left_panel.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        self.left_panel.pack_propagate(False)
        
        self.right_panel = tk.Frame(self.root, bg=self.bg_color)
        self.right_panel.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)
        
        self.right_panel.columnconfigure(0, weight=1)
        self.right_panel.rowconfigure(0, weight=1)
        self.right_panel.rowconfigure(1, weight=1)
        self.right_panel.rowconfigure(2, weight=1)
            
        self.plot_frames = []
        for i in range(3):
            frame = tk.Frame(self.right_panel, bg=self.plot_bg_color, bd=2, relief='groove')
            frame.grid(row=i, column=0, sticky='nsew', padx=5, pady=5)
            self.plot_frames.append(frame)
            
        self.waveform_plot = WaveformPlot(self.plot_frames[0])
        self.spectrogram_plot = SpectrogramPlot(self.plot_frames[1])
        self.constellation_plot = ConstellationPlot(self.plot_frames[2])
        
        self.controls = ControlPanel(self.left_panel, self)
        
        self._update_layout_for_mode()

    def _update_layout_for_mode(self):
        if self.config.mode == "morse":
            self.plot_frames[2].grid_remove()
        else:
            self.plot_frames[2].grid(row=2, column=0, sticky='nsew', padx=5, pady=5)

    def _setup_status_bar(self):
        self.status_var = tk.StringVar()
        self.status_var.set("Ready | Morse Code Transceiver")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, bg=self.accent_color, fg=self.text_color, font=("Consolas", 10))
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky='ew')

    def run(self):
        self.root.mainloop()

    def update_plots(self, tx_signal: Optional[np.ndarray] = None, rx_signal: Optional[np.ndarray] = None, **kwargs):
        sr = getattr(self.config.audio, 'sample_rate', 44100)
        
        if tx_signal is not None and rx_signal is not None:
            self.waveform_plot.plot_dual(tx_signal, rx_signal, sample_rate=sr)
        elif tx_signal is not None:
            self.waveform_plot.plot(tx_signal, sample_rate=sr, label="TX Signal", color="cyan")
        elif rx_signal is not None:
            self.waveform_plot.plot(rx_signal, sample_rate=sr, label="RX Signal", color="#e94560")

        spec_sig = rx_signal if (rx_signal is not None and len(rx_signal) > 0) else tx_signal
        if spec_sig is not None and len(spec_sig) > 0:
            self.spectrogram_plot.plot(spec_sig, sample_rate=sr)
            
        # If in OFDM mode and we have constellation data in kwargs
        if self.config.mode == "ofdm" and 'constellation' in kwargs:
            pre_eq, post_eq = kwargs['constellation']
            self.constellation_plot.plot(pre_eq, post_eq)

    def set_status(self, message: str):
        self.status_var.set(message)
        self.root.update_idletasks()

    def update_mode(self, mode: str):
        self.config.mode = mode
        self._update_layout_for_mode()
