import matplotlib.pyplot as plt
from script_signal import denoise, SIGNAL_PATH

def plot_results(result_file_path, Ms):

    pds_values = []
    psnr_values = [] 

    with open(result_file_path) as f:
        next(f)

        for line in f:
            pds, _, psnr = line.strip().split(',')

            pds_values.append(float(pds))
            psnr_values.append(float(psnr))

    plt.plot(pds_values, psnr_values, marker = 'o', linestyle = '-', color = 'b', label = 'PSNR')
    plt.xlabel('pds')
    plt.ylabel('PSNR')
    plt.title(f'PSNR vs pds for Ms = {Ms}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/figures/pds_vs_PSNR_Ms={Ms}.png')

plot_results('results/pds_PSNR_log_Ms=600.txt', 600)