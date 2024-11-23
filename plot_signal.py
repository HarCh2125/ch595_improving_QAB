import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Function to plot any signal
def plot_signal(signal, plot_path, title="Signal"):
    plt.figure()
    plt.plot(signal, label='Signal')
    plt.title(title)
    # plt.xlabel('Sample Index')
    # plt.ylabel('Amplitude')
    # plt.grid(True)
    plt.legend()
    plt.savefig(plot_path)

# Load the .mat file and extract the signal
def load_mat_file(filepath):
    mat_data = loadmat(filepath)
    signal = mat_data['S'].squeeze()  # Remove unnecessary dimensions if needed
    return signal

# Load an image (converted to an .npy file)
def load_npy_file(filepath):
    arr = np.load(filepath)
    arr_resized = np.reshape(arr, (arr.shape[0] * arr.shape[1],))
    return arr_resized

# signal_path = 'data/sample_signal.mat'
# plot_path = 'data/sample_signal_plot.png'
# signal = load_mat_file(signal_path)
# print(signal)
# print(signal.shape)

image_path = 'data/image/boat.npy'
image = load_npy_file(image_path)
print(image)
print(image.shape)


# plot_signal(signal, plot_path, title = 'Clean Signal')