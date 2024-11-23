# DESCRIPTION OF WHAT THIS CODE DOES:
# This code computes a Heaviside-like function used for thresholding.

import numpy as np

def heavi(x, eps):
    """
    Heaviside-like function used for thresholding.
    :param x: input array
    :param eps: epsilon value
    :return: heaviside-like function result
    """
    N = x.shape[0]
    fonct = np.zeros(N)

    gauche = np.ones(N)
    milieu = 0.5 * (1 - x / eps - (1 / np.pi) * np.sin(np.pi * x / eps))

    H = fonct + gauche * (x <= -eps) + milieu * ((x > -eps) & (x < eps))

    return H
