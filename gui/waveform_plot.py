from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk

class WaveformPlot:
    """
    Time-Domain Plot widget for visualizing continuous signals.
    """
    def __init__(self, parent_frame: tk.Frame, figsize: tuple = (6, 3)):
        """
        Initialize the waveform plot.
        
        Parameters
        ----------
        parent_frame : tk.Frame
            The Tkinter frame to embed the plot into.
        figsize : tuple
            Figure size in inches (width, height).
        """
        self.fig = Figure(figsize=figsize, dpi=100, facecolor='#0f3460')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#0f3460')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.line_tx = None
        self.line_rx = None
        
        self._format_axes()
        
    def _format_axes(self):
        """Format the axes colors and grid."""
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        self.ax.grid(True, color='#ffffff', alpha=0.2)
        for spine in self.ax.spines.values():
            spine.set_color('white')

    def plot(self, signal: np.ndarray, sample_rate: int = 44100, label: str = 'Signal', color: str = '#e94560'):
        """
        Plot a single time-domain waveform.
        
        Parameters
        ----------
        signal : np.ndarray
            The 1D signal array.
        sample_rate : int
            Sampling frequency in Hz.
        label : str
            Legend label.
        color : str
            Hex color code for the line.
        """
        self.clear()
        time_axis = np.arange(len(signal)) * 1000.0 / sample_rate  # in ms
        self.line_tx, = self.ax.plot(time_axis, signal, color=color, label=label, linewidth=1)
        self.ax.set_xlabel('Time (ms)')
        self.ax.set_ylabel('Amplitude')
        self.ax.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
        self.canvas.draw()

    def plot_dual(self, tx_signal: np.ndarray, rx_signal: np.ndarray, sample_rate: int = 44100):
        """
        Overlay TX and RX signals on the same plot.
        
        Parameters
        ----------
        tx_signal : np.ndarray
            Transmitted signal array.
        rx_signal : np.ndarray
            Received signal array.
        sample_rate : int
            Sampling frequency in Hz.
        """
        self.clear()
        time_axis_tx = np.arange(len(tx_signal)) * 1000.0 / sample_rate
        time_axis_rx = np.arange(len(rx_signal)) * 1000.0 / sample_rate
        
        self.line_tx, = self.ax.plot(time_axis_tx, tx_signal, color='cyan', label='TX Signal', linewidth=1, alpha=0.8)
        self.line_rx, = self.ax.plot(time_axis_rx, rx_signal, color='#e94560', label='RX Signal', linewidth=1, alpha=0.8)
        
        self.ax.set_xlabel('Time (ms)')
        self.ax.set_ylabel('Amplitude')
        self.ax.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
        self.canvas.draw()

    def clear(self):
        """Clear the plot axes."""
        self.ax.clear()
        self._format_axes()
        self.line_tx = None
        self.line_rx = None
        self.canvas.draw()

    def update(self, signal: np.ndarray):
        """
        Update the plot data without recreating the figure.
        
        Parameters
        ----------
        signal : np.ndarray
            New signal data.
        """
        if self.line_tx is not None:
            self.line_tx.set_ydata(signal)
            self.line_tx.set_xdata(np.arange(len(signal))) # Simplified update
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw()
        else:
            self.plot(signal)

    def set_title(self, title: str):
        """Set the plot title."""
        self.ax.set_title(title)
        self.canvas.draw()
