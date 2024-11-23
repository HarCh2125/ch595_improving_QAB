import sys
sys.path.insert(1, '../')
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from plot_signal import load_mat_file

# Import the signal
S = load_mat_file('data/sample_signal.mat')
N = S.shape[0]
SNR = 15
pS = np.sum(S ** 2) / N
B = np.random.randn(N) * np.sqrt(np.abs(S))  # Poisson noise
pB_tmp = np.sum(B ** 2) / N
B = B / np.sqrt(pB_tmp) * np.sqrt(pS * 10 ** (-SNR / 10))
signal = B + S

# Apply Gaussian filtering
sigma = 2  # Adjust sigma to control the amount of smoothing
filtered_signal = gaussian_filter1d(signal, sigma)

# Plot the original and filtered signals
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(signal, label='Original Signal')
plt.title('Original Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(filtered_signal, label='Filtered Signal')
plt.title('Filtered Signal (Gaussian Filter, σ={})'.format(sigma))
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()