import numpy as np
from scipy.linalg import eigh
from scipy.fft import fft, ifft, fftfreq
from plot_signal import load_mat_file
import matplotlib.pyplot as plt

# Parameters
N = 64  # Number of DVR grid points
Ms = 20  # Number of basis functions to keep
pds = 0.5  # Planck-like constant
threshold_energy = 0.1  # Energy threshold for wavefunction filtering
SIGNAL_PATH = 'data/sample_signal.mat'

# Generate a DVR grid for a Fourier basis
def dvr_grid(N, L=1.0):
    dx = L / N
    x = np.linspace(-L/2, L/2, N)
    k = fftfreq(N, d=dx) * 2 * np.pi  # Fourier DVR momentum grid
    return x, k

# Construct the Hamiltonian on the DVR grid
def construct_hamiltonian(signal, pds, k):
    N = len(signal)
    H = np.diag(signal) + np.diag(pds**2 * k**2 / 2)
    return H

# Quantum Adaptive Basis (QAB) denoising function
def qab_denoising_dvr(signal, Ms, pds, threshold_energy):
    # DVR grid and Fourier basis setup
    N = len(signal)
    x, k = dvr_grid(N)
    
    # Construct the Hamiltonian using DVR
    H = construct_hamiltonian(signal, pds, k)
    
    # Eigenvalue decomposition to get wavefunctions
    eigvals, eigvecs = eigh(H)
    
    # Filter wavefunctions based on energy threshold
    psi = []
    for i in range(N):
        if eigvals[i] < threshold_energy:
            psi.append(eigvecs[:, i])
    
    # Reconstruct the denoised signal
    denoised_signal = np.sum([np.dot(psi_i, signal) * psi_i for psi_i in psi], axis=0)
    
    return denoised_signal

# Calculate MSE and PSNR
def calculate_mse(original, denoised):
    return np.mean((original - denoised) ** 2)

def calculate_psnr(original, denoised):
    mse = calculate_mse(original, denoised)
    if mse == 0:
        return float('inf')
    max_val = np.max(original)
    return 10 * np.log10(max_val ** 2 / mse)

# Main Script
if __name__ == "__main__":
    # Load the clean data
    S = load_mat_file(SIGNAL_PATH)
    # S = data['S']  # Assuming 'S' is stored in the .npy file

    # Generate noise
    N = S.shape[0]
    SNR = 15
    pS = np.sum(S ** 2) / N
    B = np.random.randn(N) * np.sqrt(np.abs(S))  # Poisson noise
    pB_tmp = np.sum(B ** 2) / N
    B = B / np.sqrt(pB_tmp) * np.sqrt(pS * 10 ** (-SNR / 10))
    noisy_signal = B + S  # Noisy signal

    # Apply QAB denoising with DVR grid
    denoised_signal = qab_denoising_dvr(noisy_signal, Ms, pds, threshold_energy)

    # Calculate PSNR
    psnr = calculate_psnr(S, denoised_signal)

    # Plot figures
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(S)
    plt.title("Clean Signal")
    plt.axis([0, 512, -10, 25])

    plt.subplot(1, 3, 2)
    plt.plot(noisy_signal)
    plt.title(f"Noisy Signal \n(SNR = {SNR:.2f} dB)")
    plt.axis([0, 512, -10, 25])

    plt.subplot(1, 3, 3)
    plt.plot(denoised_signal)
    plt.title(f"Denoised Signal \n(PSNR = {psnr:.2f} dB)")
    plt.axis([0, 512, -10, 25])

    plt.savefig(f'results/DVR_denoising_Ms={Ms}_pds={pds}_threshE={threshold_energy}.png')