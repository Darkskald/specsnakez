import os
import pathlib
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from specsnake.base_spectrum import BaseSpectrum


# todo add support for plot file

class Plotter:
    # todo: custom title, label
    def __init__(self, specs: List[BaseSpectrum], title="default", config=None):
        self.specs = specs
        self.config = config if config is not None else {}
        self.title = title

        self.fig, self.ax = plt.subplots()

    def plot(self, show_label=True):
        for s in self.specs:
            label = s.name if show_label else None
            self.ax.plot(s.x, s.y, label=label, **self.config)
        self.ax.set_xlabel(self.specs[0].x_unit)
        self.ax.set_ylabel(self.specs[0].y_unit)
        self.ax.legend()

    def plot_custom(self, x, y,  **kwargs):
        """Add a customp plot to the current axis."""
        self.ax.plot(x, y, **kwargs)
        self.ax.legend()

    def show(self):
        self.fig.show()

    def save(self):
        self.fig.savefig(f'{self.title}.png')
