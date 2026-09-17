from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk

class ConstellationPlot:

    def __init__(self, parent_frame: tk.Frame, figsize: tuple = (4, 4)):

        self.fig = Figure(figsize=figsize, dpi=100, facecolor='#0f3460')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#0f3460')

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.scatter_rx = None
        self._format_axes()

    def _format_axes(self):

        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        self.ax.grid(True, color='#ffffff', alpha=0.3, linestyle='--')
        for spine in self.ax.spines.values():
            spine.set_color('white')
        self.ax.set_aspect('equal', 'box')

        self.ax.axhline(0, color='white', linewidth=1, alpha=0.5)
        self.ax.axvline(0, color='white', linewidth=1, alpha=0.5)

    def plot(self, symbols: np.ndarray, ideal_points: np.ndarray = None, title: str = 'Constellation'):

        self.clear()

        if ideal_points is not None:
            self.ax.scatter(np.real(ideal_points), np.imag(ideal_points), 
                            c='white', marker='+', s=100, label='Ideal', alpha=0.8)

        self.scatter_rx = self.ax.scatter(np.real(symbols), np.imag(symbols), 
                                          c='cyan', marker='.', s=10, label='Received', alpha=0.6)

        self.ax.set_xlabel('In-Phase (I)')
        self.ax.set_ylabel('Quadrature (Q)')
        self.ax.set_title(title)

        if ideal_points is not None:
            self.ax.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='white', labelcolor='white')

        limit = max(np.max(np.abs(np.real(symbols))), np.max(np.abs(np.imag(symbols)))) * 1.2
        if limit == 0 or np.isnan(limit):
            limit = 1
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)

        self.canvas.draw()

    def plot_comparison(self, before_eq: np.ndarray, after_eq: np.ndarray, ideal: np.ndarray):

        self.clear()

        self.ax.scatter(np.real(ideal), np.imag(ideal), 
                        c='white', marker='+', s=100, label='Ideal', alpha=0.8)

        self.ax.scatter(np.real(before_eq), np.imag(before_eq), 
                        c='#e94560', marker='.', s=10, label='Pre-EQ', alpha=0.3)

        self.ax.scatter(np.real(after_eq), np.imag(after_eq), 
                        c='cyan', marker='.', s=10, label='Post-EQ', alpha=0.8)

        self.ax.set_xlabel('In-Phase (I)')
        self.ax.set_ylabel('Quadrature (Q)')
        self.ax.set_title('Equalization Effect')
        self.ax.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='white', labelcolor='white')

        limit = max(np.max(np.abs(np.real(ideal))), np.max(np.abs(np.imag(ideal)))) * 1.5
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)

        self.canvas.draw()

    def clear(self):

        self.ax.clear()
        self._format_axes()
        self.scatter_rx = None
        self.canvas.draw()

    def update(self, symbols: np.ndarray):

        self.plot(symbols)

    def set_title(self, title: str):

        self.ax.set_title(title)
        self.canvas.draw()
