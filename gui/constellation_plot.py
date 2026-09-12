from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk

class ConstellationPlot:
    """
    I/Q Scatter Plot widget for visualizing complex symbol constellations.
    """
    def __init__(self, parent_frame: tk.Frame, figsize: tuple = (4, 4)):
        """
        Initialize the constellation plot.
        
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
        
        self.scatter_rx = None
        self._format_axes()
        
    def _format_axes(self):
        """Format the axes colors, grid, and aspect ratio."""
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        self.ax.grid(True, color='#ffffff', alpha=0.3, linestyle='--')
        for spine in self.ax.spines.values():
            spine.set_color('white')
        self.ax.set_aspect('equal', 'box')
        
        # Draw central crosshairs
        self.ax.axhline(0, color='white', linewidth=1, alpha=0.5)
        self.ax.axvline(0, color='white', linewidth=1, alpha=0.5)

    def plot(self, symbols: np.ndarray, ideal_points: np.ndarray = None, title: str = 'Constellation'):
        """
        Scatter plot of received symbols on the I/Q plane.
        
        Parameters
        ----------
        symbols : np.ndarray
            Complex received symbols array.
        ideal_points : np.ndarray, optional
            Ideal constellation points.
        title : str
            Plot title.
        """
        self.clear()
        
        # Plot ideal points if provided
        if ideal_points is not None:
            self.ax.scatter(np.real(ideal_points), np.imag(ideal_points), 
                            c='white', marker='+', s=100, label='Ideal', alpha=0.8)
            
        # Plot received symbols
        self.scatter_rx = self.ax.scatter(np.real(symbols), np.imag(symbols), 
                                          c='cyan', marker='.', s=10, label='Received', alpha=0.6)
        
        self.ax.set_xlabel('In-Phase (I)')
        self.ax.set_ylabel('Quadrature (Q)')
        self.ax.set_title(title)
        
        if ideal_points is not None:
            self.ax.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
            
        # Adjust limits based on data
        limit = max(np.max(np.abs(np.real(symbols))), np.max(np.abs(np.imag(symbols)))) * 1.2
        if limit == 0 or np.isnan(limit):
            limit = 1
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        
        self.canvas.draw()

    def plot_comparison(self, before_eq: np.ndarray, after_eq: np.ndarray, ideal: np.ndarray):
        """
        Overlay plot showing pre/post equalization constellations.
        
        Parameters
        ----------
        before_eq : np.ndarray
            Symbols before equalization.
        after_eq : np.ndarray
            Symbols after equalization.
        ideal : np.ndarray
            Ideal constellation points.
        """
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
        """Clear the plot axes."""
        self.ax.clear()
        self._format_axes()
        self.scatter_rx = None
        self.canvas.draw()

    def update(self, symbols: np.ndarray):
        """
        Update the plot with new symbols.
        
        Parameters
        ----------
        symbols : np.ndarray
            New complex symbols.
        """
        self.plot(symbols)

    def set_title(self, title: str):
        """Set the plot title."""
        self.ax.set_title(title)
        self.canvas.draw()
