import numpy as np
import matplotlib.pyplot as plt

n_trials = int(1e4)

Ns = np.logspace(1, 5, 10)

for N in Ns:
    rs = []

    for _ in range(n_trials):
        thetas = np.random.uniform(0, 2 * np.pi, int(N))

        xs = np.exp(1j * thetas)

        r = np.abs(np.sum(xs))

        # NORMALIZATION FACTOR
        r_norm = r / np.sqrt(N)

        rs.append(r_norm)

    plt.hist(rs, density=True)

    x = np.linspace(0, 3, 100)
    p = 2 * x * np.exp(-(x ** 2))
    plt.plot(x, p)

    plt.show()
