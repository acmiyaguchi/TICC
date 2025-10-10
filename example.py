from ticc.TICC_solver import fit
import numpy as np

if __name__ == "__main__":
    print("starting TICC example", flush=True)
    fname = "example_data.txt"
    (cluster_assignment, cluster_MRFs) = fit(input_file=fname, window_size=1, number_of_clusters=8, lambda_parameter=11e-2, beta=600, maxIters=100, threshold=2e-5,
                write_out_file=False, prefix_string="output_folder/", num_proc=1)

    print(cluster_assignment)
    np.savetxt('Results.txt', cluster_assignment, fmt='%d', delimiter=',')
