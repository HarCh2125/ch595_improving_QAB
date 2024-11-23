from script_signal import denoise, SIGNAL_PATH
from plot_results import plot_results

# Define the default parameters
Ms = 200    # Original default value: 200
pds = 0.4   # Original default value: 0.4
sg = 15     # Original default value: 15

# Start with searching for the best value of pds
pds_max = 4.0
pds_step = 0.1
n_steps = int((pds_max - pds) / pds_step)

Ms_max = 600
Ms_steps = int((Ms_max - Ms)/100)

i = 0
while i in range(Ms_steps + 1):
    pds_list = []
    PSNR_list = []

    results_file_path = f'results/pds_PSNR_log_Ms={Ms}.txt'

    # Start writing to an output file
    with open(results_file_path, 'w') as f:
        f.write(f"pds, SNR, PSNR, Ms = {Ms}\n")
        for i in range(n_steps + 1):
            SNR, PSNR_end = denoise(SIGNAL_PATH, Ms, pds, sg)
            f.write(f"{pds:.2f}, {SNR:.2f}, {PSNR_end:.2f}\n")
            pds_list.append(pds)
            PSNR_list.append(PSNR_end)
            pds += pds_step

    # # Find the pds for the maximum PSNR
    # max_index = PSNR_list.index(max(PSNR_list))
    # best_pds = pds_list[max_index]
    # print(f"Best pds: {best_pds}")
    # f.write(f"Best pds: {best_pds}")

    f.close()

    # # Plot the results
    # plt.figure()
    # plt.plot(pds_list, PSNR_list)
    # plt.xlabel('pds')
    # plt.ylabel('PSNR')
    # plt.title('PSNR vs. pds')
    # plt.grid()
    # plt.savefig('results/figures/PSNR_vs_pds.png')

    # Plot the results now
    plot_results(results_file_path, Ms)

    Ms += 100
    i += 1