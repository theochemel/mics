import pickle
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from scipy.signal import spectrogram
from tqdm import tqdm


def preprocess(filename: str):
    with open(filename, "rb") as f:
        data = pickle.load(f)

    n_sinks = data['n_sinks']
    n_sources = data['n_sources']
    waves = data['rx_wave']
    T_rx = data['T_rx']

    spectrograms = []

    print("Computing spectrograms...")
    for i in tqdm(range(len(waves))):
        sink_spectrograms = []

        for sink in range(n_sinks):
            f, t, S = spectrogram(waves[i][sink], 1 / T_rx)
            sink_spectrograms.append((f, t, S))

        spectrograms.append(sink_spectrograms)

    return spectrograms

def update_plot():
    sink_spectrograms = spectrograms[index]
    global colorbars, figlabel
    for i, ax in enumerate(axes):
        f, t, S = sink_spectrograms[i]
        ax.clear()
        pcm = ax.pcolormesh(t, f, 10 * np.log10(S), shading='gouraud', cmap='viridis')

        if colorbars[i]:
            colorbars[i].remove()
        colorbars[i] = fig.colorbar(pcm, ax=ax, label='Power/Frequency (dB/Hz)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Rx{i}')

    if figlabel:
        figlabel.remove()
    figlabel = fig.text(0.5, 0.02, f'Pose {index}', ha='center', fontsize=12)

    plt.tight_layout()
    plt.draw()

index = 0
spectrograms = []
fig = None
colorbars = []
axes = None
figlabel = None

def on_key(event):
    global index
    if event.key == "right":
        index = (index + 1) % len(spectrograms)
        update_plot()
    elif event.key == "left":
        index = (index - 1) % len(spectrograms)
        update_plot()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    spectrograms = preprocess(args.filename)
    n_sinks = len(spectrograms[0])

    fig, axes = plt.subplots(n_sinks, 1, figsize=(15, 6))
    colorbars = [None for ax in axes]

    update_plot()
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

# fig, axes = plt.subplots(2, 1, figsize=(15, 6))
#
# ax0 = axes[0]
# f, t, S = spectrogram(wave[0], 1 / T_tx)
# pcm0 = ax0.pcolormesh(t, f, 10 * np.log10(S_tx), shading='gouraud', cmap='viridis')
# fig.colorbar(pcm0, ax=ax0, label='Power/Frequency (dB/Hz)')
# ax0.set_ylabel('Frequency (Hz)')
# ax0.set_xlabel('Time (s)')
# ax0.set_title('Rx0')
#
# ax1 = axes[1]
# f, t, Sxx = spectrogram(wave[1], 1 / T_rx)
# pcm1 = ax1.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
# fig.colorbar(pcm1, ax=ax1, label='Power/Frequency (dB/Hz)')
# ax1.set_ylabel('Frequency (Hz)')
# ax1.set_xlabel('Time (s)')
# ax1.set_title('Rx1')
#
# plt.tight_layout()
# plt.show()
#
