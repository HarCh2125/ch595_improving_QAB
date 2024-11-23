import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def total_variation_denoising(noisy_signal, lambda_tv=0.1):
    """
    Performs Total Variation (TV) denoising on a 1D noisy signal.
    
    Parameters:
        noisy_signal (numpy.ndarray): The noisy signal to be denoised.
        lambda_tv (float): The regularization parameter that controls the trade-off between
                           smoothness and fidelity to the noisy signal.
    
    Returns:
        numpy.ndarray: The denoised signal.
    """
    def tv_denoising_objective(S):
        """Objective function for TV denoising."""
        fidelity_term = np.sum((S - noisy_signal) ** 2) / 2  # Fidelity to noisy signal
        tv_term = np.sum(np.abs(np.diff(S)))  # Total variation term (edge-preserving)
        return fidelity_term + lambda_tv * tv_term

    # Initial guess is the noisy signal
    result = minimize(tv_denoising_objective, noisy_signal, method='L-BFGS-B')
    return result.x

# Example Usage
if __name__ == "__main__":
    # Generate a sample clean signal (sine wave)
    t = np.linspace(0, 1, 500)  # 500 samples
    clean_signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
    
    # Add Gaussian noise to the signal
    noise = np.random.normal(0, 0.3, size=clean_signal.shape)
    noisy_signal = clean_signal + noise
    
    # Apply Total Variation denoising
    denoised_signal = total_variation_denoising(noisy_signal, lambda_tv=0.1)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, clean_signal, label="Clean Signal")
    plt.title("Clean Signal")
    plt.subplot(3, 1, 2)
    plt.plot(t, noisy_signal, label="Noisy Signal")
    plt.title("Noisy Signal")
    plt.subplot(3, 1, 3)
    plt.plot(t, denoised_signal, label="Denoised Signal", color='green')
    plt.title("Denoised Signal (Total Variation Denoising)")
    plt.tight_layout()
    plt.show()
