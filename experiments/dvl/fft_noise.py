import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm

f = 70e3
fs = 300e3

mu = 0
sigma = 1e1

n_rays = int(1e3)

n_bins = int(1e4)

n_trials = int(1e1)

speckles = []

n_bins_sweep = np.linspace(5e5, 1e6, 10)
alphas = []

for n_bins in n_bins_sweep:
    print(f"t: {n_bins / fs}")

    for _ in range(n_trials):
        ray_f = f + np.random.normal(mu, sigma, n_rays)
        ray_psi = np.random.uniform(0, 2 * np.pi, n_rays)

        t = (1 / fs) * np.arange(n_bins)
        x = np.zeros_like(t)

        for i in tqdm(range(n_rays)):
            x += np.sin((2 * np.pi * ray_f[i] * t) + ray_psi[i])

        K = 1 * len(x)

        # Compute FFT
        fft_result = np.fft.rfft(x, n=K)
        # Normalize so total energy (amplitude squared) is 1
        # the factor 2 comes from the fact that we're using rfft
        # so energy on positive side of the spectrum is doubled (?)
        fft_magnitude = np.abs(fft_result) ** 2
        freqs = np.fft.rfftfreq(K, d=1/fs)

        # print(f"delta f: {freqs[1] - freqs[0]}")

        fft_magnitude /= np.sum(fft_magnitude * (freqs[1] - freqs[0]))


        normal_pdf = scipy.stats.norm.pdf(freqs, loc=f + mu, scale=sigma)

        plt.figure(figsize=(10, 4))
        plt.plot(freqs, fft_magnitude)
        plt.title("FFT with Hanning Window")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.plot(freqs, normal_pdf)
        plt.xlim(f - 5 * sigma, f + 5 * sigma)
        plt.show()

        valid = (freqs > f - 2 * sigma) & (freqs < f + 2 * sigma)
        speckle = fft_magnitude[valid] / normal_pdf[valid]

        plt.plot(freqs[valid], speckle)
        plt.show()

        speckles.append(speckle)

    speckle = np.concatenate(speckles)

    plt.hist(speckle, bins=20, density=True)

    speckle_x = np.linspace(0, 10, 100)

    alpha = np.sqrt(np.pi) / (2 * np.mean(speckle))

    # speckle_y = alpha * (2 * (alpha * speckle_x)) * np.exp(-((alpha * speckle_x) ** 2))
    speckle_y = np.exp(-speckle_x)

    plt.axvline(x=np.sqrt(np.pi) / (2 * alpha), c="r")
    plt.axvline(x=np.mean(speckle), c="g")

    plt.plot(speckle_x, speckle_y)
    plt.title(f"{n_bins}")

    plt.show()

    alphas.append(alpha)

alphas = np.array(alphas)

plt.plot(1 / n_bins_sweep, alphas)
plt.show()
