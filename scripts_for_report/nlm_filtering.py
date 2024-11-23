import numpy as np
import matplotlib.pyplot as plt

def non_local_means(signal, patch_size=5, search_size=11, h=0.1):
    """
    Performs Non-Local Means (NLM) filtering on a 1D noisy signal.
    
    Parameters:
        signal (numpy.ndarray): The noisy signal to be denoised.
        patch_size (int): The size of the patch used for similarity comparison (must be odd).
        search_size (int): The size of the search window for finding similar patches (must be odd).
        h (float): Smoothing parameter that controls the sensitivity to differences.
    
    Returns:
        numpy.ndarray: The denoised signal.
    """
    N = len(signal)
    denoised_signal = np.zeros_like(signal)
    pad_size = patch_size // 2
    padded_signal = np.pad(signal, pad_size, mode='reflect')
    
    # Precompute Gaussian weights for the patch
    gaussian_weights = np.exp(-np.linspace(-1, 1, patch_size) ** 2)
    gaussian_weights /= gaussian_weights.sum()
    
    for i in range(N):
        # Extract the reference patch
        center_idx = i + pad_size
        ref_patch = padded_signal[center_idx - patch_size // 2:center_idx + patch_size // 2 + 1]
        ref_patch = ref_patch * gaussian_weights
        
        # Initialize weights and search window
        weights = []
        search_start = max(center_idx - search_size // 2, pad_size)  # Ensure within bounds
        search_end = min(center_idx + search_size // 2 + 1, N + pad_size)  # Ensure within bounds
        
        # Compute weights for all patches in the search window
        for j in range(search_start, search_end):
            compare_patch = padded_signal[j - patch_size // 2:j + patch_size // 2 + 1]
            if len(compare_patch) == patch_size:  # Ensure patch size matches
                compare_patch = compare_patch * gaussian_weights
                distance = np.sum((ref_patch - compare_patch) ** 2)
                weight = np.exp(-distance / h ** 2)
                weights.append((weight, padded_signal[j]))
        
        # Normalize weights and compute the denoised value
        if weights:
            weights, values = zip(*weights)
            weights = np.array(weights) / np.sum(weights)
            denoised_signal[i] = np.dot(weights, values)
    
    return denoised_signal

# Example Usage
if __name__ == "__main__":
    # Generate a sample clean signal (sine wave)
    t = np.linspace(0, 1, 500)  # 500 samples
    clean_signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
    
    # Add Gaussian noise to the signal
    noise = np.random.normal(0, 0.3, size=clean_signal.shape)
    noisy_signal = clean_signal + noise
    
    # Apply Non-Local Means filtering
    denoised_signal = non_local_means(noisy_signal, patch_size=5, search_size=21, h=0.1)
    
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
    plt.title("Denoised Signal (Non-Local Means)")
    plt.tight_layout()
    plt.show()
