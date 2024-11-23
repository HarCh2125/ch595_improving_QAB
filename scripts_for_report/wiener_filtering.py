import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
from plot_signal import load_mat_file
from calc_psnr import calc_psnr

# # Generate a noisy signal
# x = np.linspace(0, 10, 100)
# y = np.sin(x) + np.random.randn(100) * 0.5

# # Apply Wiener filtering
# y_filtered = wiener(y, mysize=5)  # Adjust mysize as needed

# # Plot the original and filtered signals
# plt.figure(figsize=(10, 6))

# plt.subplot(2, 1, 1)
# plt.plot(x, y, label='Original Signal')
# plt.title('Original Signal')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True)

# plt.subplot(2, 1, 2)
# plt.plot(x, y_filtered, label='Filtered Signal')
# plt.title('Filtered Signal (Wiener Filter, Mysize=5)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# Get the data
S = load_mat_file('data/sample_signal.mat')
# S = data['S']  # Assuming 'S' is stored in the .npy file

# Generate noise
N = S.shape[0]
SNR = 15
pS = np.sum(S ** 2) / N
B = np.random.randn(N) * np.sqrt(np.abs(S))  # Poisson noise
pB_tmp = np.sum(B ** 2) / N
B = B / np.sqrt(pB_tmp) * np.sqrt(pS * 10 ** (-SNR / 10))
SB = B + S  # Noisy signal

denoised_signal = wiener(SB, 5)

psnr = calc_psnr(S, SB)

# Plot figures
plt.figure()
plt.subplot(1, 3, 1)
plt.plot(S)
plt.title("Clean Signal")
plt.axis([0, 512, -10, 25])
# plt.axis([0, 512, -10, 25])

plt.subplot(1, 3, 2)
plt.plot(SB)
plt.title(f"Noisy Signal \n(SNR = {SNR:.2f} dB)")
plt.axis([0, 512, -10, 25])
# plt.axis([0, 512, -10, 25])

plt.subplot(1, 3, 3)
plt.plot(denoised_signal)
plt.title(f"Denoised Signal \n(PSNR = {psnr:.2f} dB)")
plt.axis([0, 512, -10, 25])
# plt.axis([0, 512, -10, 25])
plt.title('Testing Wiener filtering on the sample signal')
plt.show()