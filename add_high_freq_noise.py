import numpy as np
import matplotlib.pyplot as plt

def add_localized_high_frequency_noise(signal, frequency=50, amplitude=0.1, fraction=0.2):
    """
    Adds high-frequency sinusoidal noise localized to a specific fraction of the signal.
    
    Parameters:
        signal (numpy.ndarray): The original signal to which noise will be added.
        frequency (int): The frequency of the sinusoidal noise (number of oscillations).
        amplitude (float): The amplitude of the sinusoidal noise.
        fraction (float): The fraction of the signal (0 < fraction <= 1) to localize the noise.
    
    Returns:
        numpy.ndarray: The signal with localized high-frequency noise added.
        numpy.ndarray: The noise itself.
    """
    if signal.ndim != 1:
        raise ValueError("Input signal must be a 1D NumPy array.")
    if not (0 < fraction <= 1):
        raise ValueError("Fraction must be between 0 and 1.")
    
    # Determine the range for the noise
    N = len(signal)
    localized_length = int(N * fraction)
    start_idx = (N - localized_length) // 2  # Center the noise
    end_idx = start_idx + localized_length

    # Generate high-frequency noise localized in the selected range
    x = np.linspace(0, 2 * np.pi, localized_length)
    noise_localized = amplitude * np.sin(frequency * x)

    # Create the full noise array (zeros outside the localized range)
    noise = np.zeros_like(signal)
    noise[start_idx:end_idx] = noise_localized

    # Add the noise to the original signal
    noisy_signal = signal + noise

    return noisy_signal, noise

# Example usage
if __name__ == "__main__":
    # Generate a sample signal (a sine wave)
    t = np.linspace(0, 1, 500)  # 500 samples over 1 second
    original_signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave

    # Add localized high-frequency noise
    noisy_signal, noise = add_localized_high_frequency_noise(original_signal, frequency=50, amplitude=0.2, fraction=0.2)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, original_signal, label="Original Signal")
    plt.title("Original Signal")
    plt.subplot(3, 1, 2)
    plt.plot(t, noise, label="Localized High-Frequency Noise", color='orange')
    plt.title("Localized High-Frequency Noise")
    plt.subplot(3, 1, 3)
    plt.plot(t, noisy_signal, label="Noisy Signal", color='red')
    plt.title("Signal with Localized High-Frequency Noise")
    plt.tight_layout()
    plt.show()
