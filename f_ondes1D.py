# DESCRIPTION OF WHAT THIS CODE DOES:
# This code calculates the eigenvalues and eigenvectors based on the signal.

import numpy as np

def f_ondes1d(signal, poids):
    """
    Calculate eigenvalues and eigenvectors based on the signal.
    :param signal: input signal
    :param poids: weight
    :return: eigenvectors (psi) and eigenvalues (E)
    """
    N = signal.shape[0]
    
    psi = np.zeros((N, N))  # eigenvectors
    E = np.zeros(N)         # eigenvalues

    # Construct Hamiltonian matrix
    terme_hsm = np.ones(N) * poids
    H = np.diag(signal) + np.diag(terme_hsm) * 2 \
        - np.diag(terme_hsm[:-1], -1) - np.diag(terme_hsm[:-1], 1)
    H[0, -1] = -poids
    H[-1, 0] = -poids

    # Eigen decomposition
    valP, vectP = np.linalg.eigh(H)

    # Sort eigenvectors and eigenvalues
    for g in range(N):
        i_psi = np.argmin(valP)
        psi[:, g] = vectP[:, i_psi]
        E[g] = valP[i_psi]
        valP = np.delete(valP, i_psi)
        vectP = np.delete(vectP, i_psi, axis=1)

    return psi, E
