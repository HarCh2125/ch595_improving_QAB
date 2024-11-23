# DESCRIPTION OF WHAT THIS CODE DOES:
# This code computes the Peak Signal-to-Noise Ratio (PSNR) between two images or signals.

import numpy as np

def calc_psnr(img1, img2):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR).
    :param img1: reference image or signal
    :param img2: output image or signal
    :return: PSNR value
    """
    if img1.shape != img2.shape:
        raise ValueError("Inputs must be of the same size")

    d = np.max([img1.max(), img2.max()])
    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        return float('inf')

    psnr = 10 * np.log10(d**2 / mse)
    return psnr
