from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from typing import Optional, TYPE_CHECKING, Any

from config import AirBuddsConfig, DEFAULT_CONFIG
from core.ofdm_transceiver import OFDMTransceiver
from gui.waveform_plot import WaveformPlot
from gui.spectrogram_plot import SpectrogramPlot
from gui.constellation_plot import ConstellationPlot
from gui.controls import ControlPanel

class Dashboard:
    """
    Main Dashboard Window for AirBudds Acoustic OFDM Transceiver.
    Integrates controls and real-time visualization of DSP processes.
    """
    def __init__(self, config: Optional[AirBuddsConfig] = None):
        """
        Initialize the main dashboard.
        
        Parameters
        ----------
        config : AirBuddsConfig, optional
            Configuration parameters for the transceiver.
        """
        self.config = config if config is not None else DEFAULT_CONFIG
        self.root = tk.Tk()
        self.transceiver = OFDMTransceiver(self.config)
        
        # Color Scheme
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
        """Setup main Tk window properties."""
        self.root.title('AirBudds — Acoustic OFDM Transceiver')
        self.root.geometry('1400x900')
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=3)
        self.root.rowconfigure(0, weight=1)

    def _setup_menu(self):
        """Setup menu bar."""
        menubar = tk.Menu(self.root)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save", command=lambda: messagebox.showinfo("Info", "Save clicked"))
        file_menu.add_command(label="Load", command=lambda: messagebox.showinfo("Info", "Load clicked"))
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        mode_menu = tk.Menu(menubar, tearoff=0)
        mode_menu.add_command(label="Normal", command=lambda: self.set_status("Mode: Normal"))
        mode_menu.add_command(label="Ultrasonic", command=lambda: self.set_status("Mode: Ultrasonic"))
        mode_menu.add_command(label="RIR Sim", command=lambda: self.set_status("Mode: RIR Sim"))
        menubar.add_cascade(label="Mode", menu=mode_menu)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About AirBudds", "AirBudds OFDM Transceiver\nBy Joyanta Sutradhar & Dipbroto Karmokar Dip"))
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)

    def _setup_layout(self):
        """Setup left and right panels."""
        self.left_panel = tk.Frame(self.root, bg=self.accent_color, width=350)
        self.left_panel.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        self.left_panel.pack_propagate(False)
        
        self.right_panel = tk.Frame(self.root, bg=self.bg_color)
        self.right_panel.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)
        
        for i in range(2):
            self.right_panel.columnconfigure(i, weight=1)
            self.right_panel.rowconfigure(i, weight=1)
            
        # Left Panel - Controls
        self.controls = ControlPanel(self.left_panel, self)
        
        # Right Panel - Plots
        self.plot_frames = []
        for i in range(4):
            frame = tk.Frame(self.right_panel, bg=self.plot_bg_color, bd=2, relief='groove')
            frame.grid(row=i//2, column=i%2, sticky='nsew', padx=5, pady=5)
            self.plot_frames.append(frame)
            
        self.waveform_plot = WaveformPlot(self.plot_frames[0])
        self.spectrogram_plot = SpectrogramPlot(self.plot_frames[1])
        self.constellation_plot = ConstellationPlot(self.plot_frames[2])
        
        # Channel Response Plot as a generic waveform plot for now
        self.channel_plot = WaveformPlot(self.plot_frames[3])
        self.channel_plot.set_title("Channel Response")

    def _setup_status_bar(self):
        """Setup status bar at the bottom."""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready | BER: 0.0 | SNR: 0.0 dB | Throughput: 0 kbps | Mode: Normal")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, bg=self.accent_color, fg=self.text_color, font=("Consolas", 10))
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky='ew')

    def run(self):
        """Start the Tkinter main event loop."""
        self.root.mainloop()

    def update_plots(self, tx_signal: Optional[np.ndarray] = None, rx_signal: Optional[np.ndarray] = None, 
                     constellation: Optional[np.ndarray] = None, channel_H: Optional[np.ndarray] = None, 
                     ideal_points: Optional[np.ndarray] = None):
        """
        Refresh all plot widgets with new data.
        """
        sr = getattr(self.config.audio, 'sample_rate', 44100)
        
        # 1. Waveform plot
        if tx_signal is not None and rx_signal is not None:
            self.waveform_plot.plot_dual(tx_signal, rx_signal, sample_rate=sr)
        elif tx_signal is not None:
            self.waveform_plot.plot(tx_signal, sample_rate=sr, label="TX Signal", color="cyan")
        elif rx_signal is not None:
            self.waveform_plot.plot(rx_signal, sample_rate=sr, label="RX Signal", color="#e94560")

        # 2. Spectrogram plot
        spec_sig = rx_signal if (rx_signal is not None and len(rx_signal) > 0) else tx_signal
        if spec_sig is not None and len(spec_sig) > 0:
            self.spectrogram_plot.plot(spec_sig, sample_rate=sr)

        # 3. Constellation plot
        if constellation is not None and len(constellation) > 0:
            self.constellation_plot.plot(constellation, ideal_points=ideal_points)

        # 4. Channel plot
        if channel_H is not None and len(channel_H) > 0:
            mag = 20 * np.log10(np.abs(channel_H) + 1e-10)
            self.channel_plot.clear()
            self.channel_plot.plot(mag, sample_rate=sr, label="|H[k]|", color="#00d2ff")
            self.channel_plot.set_title("Channel Frequency Response |H[k]| (dB)")

    def set_status(self, message: str):
        """Update the status bar text."""
        self.status_var.set(message)

    def get_transceiver(self) -> Any:
        """Return the OFDMTransceiver instance."""
        return self.transceiver
