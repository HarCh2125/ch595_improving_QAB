# DESCRIPTION: This function denoises a signal using Quantum Adaptive Basis (QAB).

import numpy as np
from scipy.fftpack import fft, ifft
from f_ondes1D import f_ondes1d
from heavi import heavi

def signal_denoising_qab(S, SB, Ms, pds, sg):
    """
    Denoising using Quantum Adaptive Basis (QAB).
    :param S: Clean signal
    :param SB: Noisy signal
    :param Ms: Number of iterations for reconstruction
    :param pds: Planck's constant value
    :param sg: Gaussian variance (smoothing)
    :return: Denoised signal
    """
    N = S.shape[0]
    saut = 2  # Threshold displacement
    pS = np.sum(S ** 2) / N

    Vs = np.linspace(1, 10, Ms)
    Vs = 2 ** Vs

    # These variables store values associated with the maximums
    V_ms = 0.01
    seuil_m = 1
    RSB_ms = 0  # maximum SNR for adaptive transformation
    S_s_m = np.zeros(N)

    # Gaussian smoothing
    x = np.arange(-N/2, N/2)
    x = np.concatenate([x[int(N/2):], x[:int(N/2)]])
    y = (1 / np.sqrt(2 * np.pi * sg)) * np.exp(-x ** 2 / (2 * sg))

    SB_f = fft(SB)
    y_f = fft(y)
    SB_flou = np.real(ifft(SB_f * y_f))

    # Reconstruction of signal
    psi, E = f_ondes1d(SB_flou, pds)
    a = np.linalg.solve(psi, SB)

    for compteur, v in enumerate(Vs):
        print(f"Threshold slope parameter: {compteur+1}/{Ms} - Planck = {pds:.1f}")

        for k in range(1, N, saut):
            taux = heavi(np.arange(1, N + 1) - k + 2, v)

            n_SB = np.sum([taux[j] * a[j] * psi[:, j] for j in range(N)], axis=0)

            pnB = np.sum((n_SB - S) ** 2) / N
            RSB_n = 10 * np.log10(pS / pnB)

            if RSB_ms < RSB_n:
                V_ms = v
                seuil_m = k
                S_s_m = n_SB
                RSB_ms = RSB_n

    return S_s_m
