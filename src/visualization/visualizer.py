import pickle
from argparse import ArgumentParser
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from tqdm import tqdm



class SpectrogramVisualizer:
    def __init__(self):
        self._index = 0
        self._spectrograms = []
        self._fig = None
        self._colorbars = []
        self._axes = None
        self._figlabel = None

    def initialize(self, filename: str):
        with open(filename, "rb") as f:
            simulation_results = pickle.load(f)

        self._n_sinks: int = 2 # simulation_results['n_sinks']
        self._rx_pattern: List[np.array] = simulation_results['rx_pattern']
        self._rx_pattern = [p[(0, -1), :] for p in self._rx_pattern]
        self._T_rx: float = simulation_results['T_rx']

        self._compute_spectrograms()

        self._fig, self._axes = plt.subplots(self._n_sinks, 1, figsize=(15, 6))
        self._colorbars = [None for ax in self._axes]

        self._update_plot()
        self._fig.canvas.mpl_connect("key_press_event", self.on_key)
        plt.show()

    def _compute_spectrograms(self):
        self._spectrograms = []

        print("Computing spectrograms...")
        for i in tqdm(range(len(self._rx_pattern))):
            sink_spectrograms = []

            for sink in range(self._n_sinks):
                f, t, S = spectrogram(self._rx_pattern[i][sink], 1 / self._T_rx)
                sink_spectrograms.append((f, t, S))

            self._spectrograms.append(sink_spectrograms)

    def _update_plot(self):
        sink_spectrograms = self._spectrograms[self._index]
        for i, ax in enumerate(self._axes):
            f, t, S = sink_spectrograms[i]
            ax.clear()
            pcm = ax.pcolormesh(t, f, 10 * np.log10(S), shading='gouraud', cmap='viridis')

            if self._colorbars[i]:
                self._colorbars[i].remove()
            self._colorbars[i] = self._fig.colorbar(pcm, ax=ax, label='Power/Frequency (dB/Hz)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'Rx{i}')

        if self._figlabel:
            self._figlabel.remove()
        self._figlabel = self._fig.text(0.5, 0.02, f'Pose {self._index}', ha='center', fontsize=12)

        plt.tight_layout()
        plt.draw()


    def on_key(self, event):
        if event.key == "right":
            self._index = (self._index + 1) % len(self._spectrograms)
            self._update_plot()
        elif event.key == "left":
            self._index = (self._index - 1) % len(self._spectrograms)
            self._update_plot()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    sv = SpectrogramVisualizer()
    sv.initialize(args.filename)
