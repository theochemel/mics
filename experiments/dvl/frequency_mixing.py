import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

deltas = np.linspace(-0.0,  1.0, 100)
amplitudes = []

n_trials = int(1e4)
N = int(1e3)

k = N // 4
i = np.arange(N)

for delta in tqdm(deltas):
    trial_amplitudes = []

    for _ in range(n_trials):

        x1 = np.real(np.exp(2j * np.pi * ((k + 0.0) / N) * i + 1j * np.random.uniform(0, 2 * np.pi)))
        x2 = np.real(np.exp(2j * np.pi * ((k + delta) / N) * i + 1j * np.random.uniform(0, 2 * np.pi)))

        x = x1 + x2

        fft = np.fft.rfft(x)
        fft_amplitude = np.abs(fft) / N

        trial_amplitudes.append(fft_amplitude[k])

    trial_amplitudes = np.array(trial_amplitudes)

    amplitudes.append(np.mean(trial_amplitudes))

plt.plot(deltas, amplitudes)
plt.show()