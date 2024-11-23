import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from plot_signal import load_mat_file

def f_ondes1D(signal, poids):
    """
    Compute the eigenvectors (eigenstates) and eigenvalues of the Hamiltonian matrix.
    """
    N = len(signal)
    psi = np.zeros((N, N))  # Eigenvectors
    E = np.zeros(N)         # Eigenvalues

    # Construct the Hamiltonian matrix
    terme_hsm = np.ones(N) * poids
    H = np.diag(signal) + np.diag(terme_hsm) * 2 \
        - np.diag(terme_hsm[:-1], -1) - np.diag(terme_hsm[:-1], 1)
    H[0, -1] = -poids
    H[-1, 0] = -poids

    # Compute eigenvalues and eigenvectors
    eig_vals, eig_vecs = eigh(H)
    
    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eig_vals)
    eig_vals = eig_vals[sorted_indices]
    eig_vecs = eig_vecs[:, sorted_indices]
    
    return eig_vecs, eig_vals

# Generate synthetic signal
signal = load_mat_file('data/sample_signal.mat')

# Parameters for Hamiltonian
poids = 20.5

# Compute eigenstates and eigenvalues
eigenstates, eigenvalues = f_ondes1D(signal, poids)

# Plot the first 5 eigenstates
plt.figure(figsize=(10, 6))
for i in range(5):  # Plot the first 5 eigenstates
    plt.plot(np.linspace(0, 1, len(signal)), eigenstates[:, i], label=f'Eigenstate {i+1} (E={eigenvalues[i]:.2f})')

plt.title('First 5 Eigenstates of the Synthetic Signal')
plt.xlabel('Signal Domain')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()