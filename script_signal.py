# DESCRIPTION OF WHAT THIS CODE DOES:
# This code denoises a signal using the Quantum Adaptive Basis (QAB) method.

import numpy as np
import matplotlib.pyplot as plt
import os

from calc_psnr import calc_psnr
from signal_denoising_QAB import signal_denoising_qab
from plot_signal import *

SIGNAL_PATH = 'data/sample_signal.mat'
# PLOT_PATH = 'results/other_results'
PLOT_PATH = 'scripts_for_report/plots/6'
RESULTS_PATH = 'scripts_for_report/results/optimal_6.txt'
# RESULTS_PATH = 'results/image_denoise_test_1.txt'
IMAGE_PATH = 'data/image/boat.npy'

def denoise(signal_path = SIGNAL_PATH, Ms = 200, pds = 0.4, sg = 15, plot_path = PLOT_PATH, results_path = RESULTS_PATH, image_path = IMAGE_PATH):
    """
    Denoise a signal using the Quantum Adaptive Basis (QAB) method.
    Parameters:
    - signal_path (str): path to the signal file
    - Ms (int): number of iterations
    - pds (float): Planck's constant
    - sg (float): Gaussian variance (smoothing)
    """

    # Load data
    # S = load_npy_file(image_path)
    S = load_mat_file(signal_path)
    # S = data['S']  # Assuming 'S' is stored in the .npy file

    # Generate noise
    N = S.shape[0]
    SNR = 15
    pS = np.sum(S ** 2) / N
    B = np.random.randn(N) * np.sqrt(np.abs(S))  # Poisson noise
    pB_tmp = np.sum(B ** 2) / N
    B = B / np.sqrt(pB_tmp) * np.sqrt(pS * 10 ** (-SNR / 10))
    SB = B + S  # Noisy signal

    # Signal denoising using QAB
    S_result = signal_denoising_qab(S, SB, Ms, pds, sg)

    # Calculate SNR and PSNR
    pnB = np.sum((S_result - S) ** 2) / len(S)
    SNR_end = 10 * np.log10(pS / pnB)
    PSNR_end = calc_psnr(S, S_result)

    # Print data
    with open(results_path, 'a') as file:
        file.write(f"OUTPUT:\n SNR = {SNR_end:.2f} and PSNR = {PSNR_end:.2f}\n")

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
    plt.plot(S_result)
    plt.title(f"Denoised Signal \n(PSNR = {PSNR_end:.2f} dB)")
    plt.axis([0, 512, -10, 25])
    # plt.axis([0, 512, -10, 25])

    final_plot_path = os.path.join(plot_path, f'denoised_image_Ms={Ms}_pds={pds}_sg={sg}_PSNR={PSNR_end}')

    # plt.tight_layout()
    plt.savefig(f'{final_plot_path}.png')

    return SNR, PSNR_end

# denoise(Ms = 200, pds = 50)
# denoise(Ms = 700, pds = 50, sg = 20)
# denoise(Ms = 700, pds = 50, sg = 30)
# denoise(Ms = 700, pds = 20.5, sg = 100)
# denoise(Ms = 5000, pds = 25, sg = 20)
# denoise(Ms = 1, pds = 0.1, sg = 0.01)
# denoise(Ms = 500)
# denoise(sg = 5)
# denoise(sg = 10)
# denoise(sg = 20)
denoise(Ms = 800, pds = 20.5, sg = 20)