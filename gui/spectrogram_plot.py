from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk

class SpectrogramPlot:

    def __init__(self, parent_frame: tk.Frame, figsize: tuple = (6, 3)):

        self.fig = Figure(figsize=figsize, dpi=100, facecolor='#0f3460')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#0f3460')

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.colorbar = None
        self._format_axes()

    def _format_axes(self):

        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        for spine in self.ax.spines.values():
            spine.set_color('white')

    def plot(self, signal: np.ndarray, sample_rate: int = 44100, nfft: int = 1024):

        self.clear()

        _, _, _, im = self.ax.specgram(
            signal, 
            NFFT=nfft, 
            Fs=sample_rate, 
            noverlap=nfft//2, 
            cmap='viridis',
            scale='dB'
        )

        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Frequency (Hz)')

        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(im, ax=self.ax)
            self.colorbar.ax.yaxis.set_tick_params(color='white')
            self.colorbar.outline.set_edgecolor('white')
            plt.setp(plt.getp(self.colorbar.ax.axes, 'yticklabels'), color='white')
            self.colorbar.set_label('Power (dB)', color='white')
        else:
            self.colorbar.update_normal(im)

        self.canvas.draw()

    def clear(self):

        self.ax.clear()
        self._format_axes()
        self.canvas.draw()

    def update(self, signal: np.ndarray, sample_rate: int = 44100):

        self.plot(signal, sample_rate=sample_rate)

    def set_title(self, title: str):

        self.ax.set_title(title)
        self.canvas.draw()

    def set_frequency_range(self, f_min: float = 0, f_max: float = 22050):

        self.ax.set_ylim(f_min, f_max)
        self.canvas.draw()
