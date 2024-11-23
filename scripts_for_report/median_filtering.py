import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

# Generate a noisy signal
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.randn(100) * 0.5

# Apply median filtering
y_filtered = medfilt(y, kernel_size=5)  # Adjust kernel_size as needed

# Plot the original and filtered signals in separate subplots
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(x, y, label='Original Signal')
plt.title('Original Signal')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(x, y_filtered, label='Filtered Signal')
plt.title('Filtered Signal (Median Filter, Kernel Size=5)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()