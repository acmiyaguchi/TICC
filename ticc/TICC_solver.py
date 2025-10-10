"""
This file contains a functional implementation of the TICC solver.
"""

import numpy as np
import math, time, collections, os, errno, sys, code, random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gmmx
import faiss
import pandas as pd
import multiprocessing as mp

from ticc.TICC_helper import (
    getTrainTestSplit,
    upperToFull,
    updateClusters,
    find_matching,
    compute_confusion_matrix,
    computeBIC,
)
from ticc.admm_solver import admm_solver
import jax.numpy as jnp
import jax
import tqdm


def solve_all_clusters_vmap(cluster_data_list, lambda_parameter, window_size):
    """
    Batch process multiple clusters using JAX vmap for parallel computation.

    Args:
        cluster_data_list: List of empirical covariance matrices for each cluster
        lambda_parameter: Regularization parameter
        window_size: Window size for temporal blocks

    Returns:
        solutions: List of solutions for each cluster
        convergence_status: List of convergence status for each cluster
    """
    # Convert list to JAX arrays for batch processing
    if not cluster_data_list:
        return [], []

    # Get dimensions from first covariance matrix
    probSize = cluster_data_list[0].shape[0]
    size_blocks = probSize // window_size

    # Create batch of lambda matrices
    lamb_matrices = []
    for S in cluster_data_list:
        lamb = jnp.zeros((probSize, probSize)) + lambda_parameter
        lamb_matrices.append(lamb)

    # Stack the inputs for batch processing
    S_batch = jnp.stack([jnp.array(S) for S in cluster_data_list])
    lamb_batch = jnp.stack(lamb_matrices)

    # Import the JIT version directly to avoid status string conversion issues
    from ticc.jax_admm_solver import admm_solver_jit
    
    # Create vectorized version of the JAX ADMM solver JIT function
    batch_solver = jax.vmap(
        admm_solver_jit,
        in_axes=(0, 0, None, None, None, None, None, None),
        out_axes=(0, 0),
    )

    # Solve all clusters in parallel
    solutions, convergence_flags = batch_solver(
        S_batch,
        lamb_batch,
        window_size,  # num_stacked
        size_blocks,  # size_blocks
        1.0,  # rho
        1000,  # maxIters
        1e-6,  # eps_abs
        1e-6,  # eps_rel
    )

    # Convert back to Python lists and create status strings
    solutions_list = [np.array(sol) for sol in solutions]
    status_list = ["Optimal" if bool(flag) else "Incomplete: max iterations reached" 
                   for flag in convergence_flags]

    return solutions_list, status_list


def log_parameters(lambda_parameter, switch_penalty, number_of_clusters, window_size):
    print("lam_sparse", lambda_parameter)
    print("switch_penalty", switch_penalty)
    print("num_cluster", number_of_clusters)
    print("num stacked", window_size)


def load_data(input_file):
    df = pd.read_csv(input_file)
    Data = df.values
    (m, n) = Data.shape  # m: num of observations, n: size of observation vector
    print("completed getting the data")
    return Data, m, n


def prepare_out_directory(prefix_string, lambda_parameter, number_of_clusters):
    str_NULL = (
        prefix_string
        + "lam_sparse="
        + str(lambda_parameter)
        + "maxClusters="
        + str(number_of_clusters + 1)
        + "/"
    )
    if not os.path.exists(os.path.dirname(str_NULL)):
        try:
            os.makedirs(os.path.dirname(str_NULL))
        except OSError as exc:  # Guard against race condition of path already existing
            if exc.errno != errno.EEXIST:
                raise

    return str_NULL


def stack_training_data(Data, n, num_train_points, training_indices, window_size):
    complete_D_train = np.zeros([num_train_points, window_size * n])
    for i in range(num_train_points):
        for k in range(window_size):
            if i + k < num_train_points:
                idx_k = training_indices[i + k]
                complete_D_train[i][k * n : (k + 1) * n] = Data[idx_k][0:n]
    return complete_D_train


def optimize_clusters_multiprocessing(
    cluster_mean_info,
    cluster_mean_stacked_info,
    complete_D_train,
    empirical_covariances,
    len_train_clusters,
    n,
    pool,
    train_clusters_arr,
    number_of_clusters,
    window_size,
    lambda_parameter,
    biased,
    computed_covariance,
    train_cluster_inverse,
):
    """Combined function that does both training and optimization using multiprocessing."""
    # Part 1: Train clusters (from old train_clusters function)
    optRes = [None for i in range(number_of_clusters)]
    for cluster in range(number_of_clusters):
        cluster_length = len_train_clusters[cluster]
        if cluster_length != 0:
            size_blocks = n
            indices = train_clusters_arr[cluster]
            D_train = np.zeros([cluster_length, window_size * n])
            for i in range(cluster_length):
                point = indices[i]
                D_train[i, :] = complete_D_train[point, :]

            cluster_mean_info[number_of_clusters, cluster] = np.mean(D_train, axis=0)[
                (window_size - 1) * n : window_size * n
            ].reshape([1, n])
            cluster_mean_stacked_info[number_of_clusters, cluster] = np.mean(
                D_train, axis=0
            )
            ##Fit a model - OPTIMIZATION
            probSize = window_size * size_blocks
            lamb = np.zeros((probSize, probSize)) + lambda_parameter
            S = np.cov(np.transpose(D_train), bias=biased)
            empirical_covariances[cluster] = S

            # apply to process pool
            optRes[cluster] = pool.apply_async(
                admm_solver,
                (
                    S,
                    lamb,
                    window_size,
                    size_blocks,
                    1,  # rho
                    1000,
                    1e-6,
                    1e-6,
                    False,
                ),
            )

    # Part 2: Optimize clusters (from old optimize_clusters function)
    for cluster in range(number_of_clusters):
        if optRes[cluster] is None:
            continue
        val, status = optRes[cluster].get()
        print("OPTIMIZATION for Cluster #", cluster, "DONE!!!")
        # THIS IS THE SOLUTION
        S_est = upperToFull(val, 0)
        X2 = S_est
        u, _ = np.linalg.eig(S_est)
        cov_out = np.linalg.inv(X2)

        # Store the log-det, covariance, inverse-covariance, cluster means, stacked means
        computed_covariance[number_of_clusters, cluster] = cov_out
        train_cluster_inverse[cluster] = X2

    for cluster in range(number_of_clusters):
        print(
            "length of the cluster ",
            cluster,
            "------>",
            len_train_clusters[cluster],
        )


def optimize_clusters_jax(
    cluster_mean_info,
    cluster_mean_stacked_info,
    complete_D_train,
    empirical_covariances,
    len_train_clusters,
    n,
    train_clusters_arr,
    number_of_clusters,
    window_size,
    lambda_parameter,
    biased,
    computed_covariance,
    train_cluster_inverse,
):
    """JAX adapter function that prepares data and processes results."""
    # Create empty list for cluster data
    cluster_data_list = []
    cluster_indices = []  # Track which clusters have data

    # Loop to prepare D_train arrays for each cluster
    for cluster in range(number_of_clusters):
        cluster_length = len_train_clusters[cluster]
        if cluster_length != 0:
            indices = train_clusters_arr[cluster]
            D_train = np.zeros([cluster_length, window_size * n])
            for i in range(cluster_length):
                point = indices[i]
                D_train[i, :] = complete_D_train[point, :]

            # Store cluster means
            cluster_mean_info[number_of_clusters, cluster] = np.mean(D_train, axis=0)[
                (window_size - 1) * n : window_size * n
            ].reshape([1, n])
            cluster_mean_stacked_info[number_of_clusters, cluster] = np.mean(
                D_train, axis=0
            )

            # Compute empirical covariance
            S = np.cov(np.transpose(D_train), bias=biased)
            empirical_covariances[cluster] = S

            # Add to cluster data list for batch processing
            cluster_data_list.append(S)
            cluster_indices.append(cluster)

    # Call JAX batch solver
    if cluster_data_list:
        solutions, convergence_status = solve_all_clusters_vmap(
            cluster_data_list, lambda_parameter, window_size
        )

        # Process results and update dictionaries
        for i, cluster in enumerate(cluster_indices):
            val = solutions[i]
            print("OPTIMIZATION for Cluster #", cluster, "DONE!!!")

            # THIS IS THE SOLUTION
            S_est = upperToFull(val, 0)
            X2 = S_est
            u, _ = np.linalg.eig(S_est)
            cov_out = np.linalg.inv(X2)

            # Store the covariance, inverse-covariance
            computed_covariance[number_of_clusters, cluster] = cov_out
            train_cluster_inverse[cluster] = X2

    for cluster in range(number_of_clusters):
        print(
            "length of the cluster ",
            cluster,
            "------>",
            len_train_clusters[cluster],
        )


def smoothen_clusters(
    cluster_mean_info,
    computed_covariance,
    cluster_mean_stacked_info,
    complete_D_train,
    n,
    number_of_clusters,
    num_blocks,
):
    clustered_points_len = len(complete_D_train)
    inv_cov_dict = {}  # cluster to inv_cov
    log_det_dict = {}  # cluster to log_det
    for cluster in range(number_of_clusters):
        cov_matrix = computed_covariance[(number_of_clusters, cluster)][
            0 : (num_blocks - 1) * n, 0 : (num_blocks - 1) * n
        ]

        inv_cov_matrix = np.linalg.inv(cov_matrix)
        log_det_cov = np.log(np.linalg.det(cov_matrix))  # log(det(sigma2|1))
        inv_cov_dict[cluster] = inv_cov_matrix
        log_det_dict[cluster] = log_det_cov
    # For each point compute the LLE
    print("beginning the smoothening ALGORITHM")
    LLE_all_points_clusters = np.zeros([clustered_points_len, number_of_clusters])
    for point in range(clustered_points_len):
        if point + num_blocks - 2 < complete_D_train.shape[0]:
            for cluster in range(number_of_clusters):
                cluster_mean_stacked = cluster_mean_stacked_info[
                    (number_of_clusters, cluster)
                ]
                x = (
                    complete_D_train[point, :]
                    - cluster_mean_stacked[0 : (num_blocks - 1) * n]
                )
                inv_cov_matrix = inv_cov_dict[cluster]
                log_det_cov = log_det_dict[cluster]
                lle = (
                    np.dot(
                        x.reshape([1, (num_blocks - 1) * n]),
                        np.dot(
                            inv_cov_matrix,
                            x.reshape([n * (num_blocks - 1), 1]),
                        ),
                    )
                    + log_det_cov
                )
                LLE_all_points_clusters[point, cluster] = lle

    return LLE_all_points_clusters


def write_plot(
    clustered_points,
    str_NULL,
    training_indices,
    number_of_clusters,
    write_out_file,
    lambda_parameter,
    switch_penalty,
):
    # Save a figure of segmentation
    plt.figure()
    plt.plot(training_indices[0 : len(clustered_points)], clustered_points, color="r")
    plt.ylim((-0.5, number_of_clusters + 0.5))
    if write_out_file:
        plt.savefig(
            str_NULL
            + "TRAINING_EM_lam_sparse="
            + str(lambda_parameter)
            + "switch_penalty = "
            + str(switch_penalty)
            + ".jpg"
        )
    plt.close("all")
    print("Done writing the figure")


def compute_matches(
    train_confusion_matrix_EM,
    train_confusion_matrix_GMM,
    train_confusion_matrix_kmeans,
    number_of_clusters,
):
    matching_Kmeans = find_matching(train_confusion_matrix_kmeans)
    matching_GMM = find_matching(train_confusion_matrix_GMM)
    matching_EM = find_matching(train_confusion_matrix_EM)
    correct_e_m = 0
    correct_g_m_m = 0
    correct_k_means = 0
    for cluster in range(number_of_clusters):
        matched_cluster_e_m = matching_EM[cluster]
        matched_cluster_g_m_m = matching_GMM[cluster]
        matched_cluster_k_means = matching_Kmeans[cluster]

        correct_e_m += train_confusion_matrix_EM[cluster, matched_cluster_e_m]
        correct_g_m_m += train_confusion_matrix_GMM[cluster, matched_cluster_g_m_m]
        correct_k_means += train_confusion_matrix_kmeans[
            cluster, matched_cluster_k_means
        ]
    return matching_EM, matching_GMM, matching_Kmeans


def compute_f_score(
    matching_EM,
    matching_GMM,
    matching_Kmeans,
    train_confusion_matrix_EM,
    train_confusion_matrix_GMM,
    train_confusion_matrix_kmeans,
    number_of_clusters,
):
    f1_EM_tr = -1
    f1_GMM_tr = -1
    f1_kmeans_tr = -1
    print("\n\n")
    print("TRAINING F1 score:", f1_EM_tr, f1_GMM_tr, f1_kmeans_tr)
    correct_e_m = 0
    correct_g_m_m = 0
    correct_k_means = 0
    for cluster in range(number_of_clusters):
        matched_cluster__e_m = matching_EM[cluster]
        matched_cluster__g_m_m = matching_GMM[cluster]
        matched_cluster__k_means = matching_Kmeans[cluster]

        correct_e_m += train_confusion_matrix_EM[cluster, matched_cluster__e_m]
        correct_g_m_m += train_confusion_matrix_GMM[cluster, matched_cluster__g_m_m]
        correct_k_means += train_confusion_matrix_kmeans[
            cluster, matched_cluster__k_means
        ]


def predict_clusters(
    trained_model, switch_penalty, num_blocks, window_size, test_data=None
):
    if test_data is not None:
        if not isinstance(test_data, np.ndarray):
            raise TypeError("input must be a numpy array!")
    else:
        test_data = trained_model["complete_D_train"]

    lle_all_points_clusters = smoothen_clusters(
        trained_model["cluster_mean_info"],
        trained_model["computed_covariance"],
        trained_model["cluster_mean_stacked_info"],
        test_data,
        trained_model["time_series_col_size"],
        trained_model["number_of_clusters"],
        num_blocks,
    )

    # Update cluster points - using NEW smoothening
    clustered_points = updateClusters(
        lle_all_points_clusters, switch_penalty=switch_penalty
    )

    return clustered_points


def fit(
    input_file,
    window_size=10,
    number_of_clusters=5,
    lambda_parameter=11e-2,
    beta=400,
    maxIters=1000,
    threshold=2e-5,
    write_out_file=False,
    prefix_string="",
    num_proc=1,
    compute_BIC=False,
    cluster_reassignment=20,
    biased=False,
):
    assert maxIters > 0

    log_parameters(lambda_parameter, beta, number_of_clusters, window_size)

    start = time.time()
    times_series_arr, time_series_rows_size, time_series_col_size = load_data(
        input_file
    )
    end = time.time()
    print("loading data took:", end - start)

    num_blocks = window_size + 1
    str_NULL = prepare_out_directory(
        prefix_string, lambda_parameter, number_of_clusters
    )

    training_indices = getTrainTestSplit(time_series_rows_size, num_blocks, window_size)
    num_train_points = len(training_indices)

    start = time.time()
    complete_D_train = stack_training_data(
        times_series_arr,
        time_series_col_size,
        num_train_points,
        training_indices,
        window_size,
    )
    end = time.time()
    print("stacking data took:", end - start)

    start = time.time()
    d = complete_D_train.shape[1]
    gmm = gmmx.GaussianMixtureModelJax.create(
        n_components=number_of_clusters, n_features=d
    )
    fitter = gmmx.EMFitter(tol=1e-3, max_iter=100)
    res = fitter.fit(x=complete_D_train, gmm=gmm)
    clustered_points = np.asarray(res.gmm.predict(complete_D_train)).flatten()
    end = time.time()
    print(f"GMM initialization took {end - start:.4f} seconds")
    gmm_clustered_pts = clustered_points.copy()

    start = time.time()
    kmeans = faiss.Kmeans(d, number_of_clusters, niter=20, verbose=False, gpu=False)
    kmeans.train(complete_D_train.astype(np.float32))
    _, clustered_points_kmeans = kmeans.index.search(
        complete_D_train.astype(np.float32), 1
    )
    kmeans_clustered_pts = np.asarray(clustered_points_kmeans).flatten()
    end = time.time()
    print(f"K-means initialization took {end - start:.4f} seconds")

    # Simplified state dictionaries
    train_cluster_inverse = {}
    computed_covariance = {}
    cluster_mean_info = {}
    cluster_mean_stacked_info = {}
    empirical_covariances = {}
    old_clustered_points = None

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=num_proc)
    for iters in tqdm.tqdm(range(maxIters)):
        train_clusters_arr = collections.defaultdict(list)
        for point, cluster_num in enumerate(clustered_points):
            train_clusters_arr[cluster_num].append(point)

        len_train_clusters = {
            k: len(train_clusters_arr[k]) for k in range(number_of_clusters)
        }

        # optimize_clusters_multiprocessing(
        #     cluster_mean_info,
        #     cluster_mean_stacked_info,
        #     complete_D_train,
        #     empirical_covariances,
        #     len_train_clusters,
        #     time_series_col_size,
        #     pool,
        #     train_clusters_arr,
        #     number_of_clusters,
        #     window_size,
        #     lambda_parameter,
        #     biased,
        #     computed_covariance,
        #     train_cluster_inverse,
        # )

        # Use JAX solver instead
        optimize_clusters_jax(
            cluster_mean_info,
            cluster_mean_stacked_info,
            complete_D_train,
            empirical_covariances,
            len_train_clusters,
            time_series_col_size,
            train_clusters_arr,
            number_of_clusters,
            window_size,
            lambda_parameter,
            biased,
            computed_covariance,
            train_cluster_inverse,
        )

        # Ensure all clusters have parameters, even if they are empty.
        # This prevents KeyErrors in the prediction step.
        if (
            len(computed_covariance) > 0
            and len(computed_covariance) < number_of_clusters
        ):
            valid_key = next(iter(computed_covariance))
            valid_cov = computed_covariance[valid_key]
            valid_mean = cluster_mean_info[valid_key]
            valid_stacked_mean = cluster_mean_stacked_info[valid_key]
            for cluster_num in range(number_of_clusters):
                key = (number_of_clusters, cluster_num)
                if key not in computed_covariance:
                    computed_covariance[key] = valid_cov
                    cluster_mean_info[key] = valid_mean
                    cluster_mean_stacked_info[key] = valid_stacked_mean

        print("OPTIMIZATION OF CLUSTERS COMPLETE")

        # The trained_model is now much simpler
        trained_model = {
            "cluster_mean_info": cluster_mean_info,
            "computed_covariance": computed_covariance,
            "cluster_mean_stacked_info": cluster_mean_stacked_info,
            "complete_D_train": complete_D_train,
            "time_series_col_size": time_series_col_size,
            "number_of_clusters": number_of_clusters,
        }
        clustered_points = predict_clusters(
            trained_model, beta, num_blocks, window_size
        )

        new_train_clusters = collections.defaultdict(list)
        for point, cluster in enumerate(clustered_points):
            new_train_clusters[cluster].append(point)

        len_new_train_clusters = {
            k: len(new_train_clusters[k]) for k in range(number_of_clusters)
        }

        # Empty cluster reassignment logic
        if iters > 0 and 0 in len_new_train_clusters.values():
            cluster_norms = [
                (np.linalg.norm(computed_covariance.get((number_of_clusters, i), 0)), i)
                for i in range(number_of_clusters)
            ]
            norms_sorted = sorted(cluster_norms, reverse=True)
            valid_clusters = [
                cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] > 0
            ]

            if not valid_clusters:
                continue  # Avoids crash if all clusters are empty

            counter = 0
            for cluster_num in range(number_of_clusters):
                if len_new_train_clusters[cluster_num] == 0:
                    cluster_selected = valid_clusters[counter % len(valid_clusters)]
                    counter += 1

                    print(
                        f"Reassigning points to empty cluster {cluster_num} from cluster {cluster_selected}"
                    )

                    # Find a point in the largest cluster to seed the reassignment
                    points_in_selected_cluster = new_train_clusters[cluster_selected]
                    if not points_in_selected_cluster:
                        continue
                    start_point = np.random.choice(points_in_selected_cluster)

                    for i in range(cluster_reassignment):
                        point_to_move = start_point + i
                        if point_to_move >= len(clustered_points):
                            break

                        clustered_points[point_to_move] = cluster_num

                        # Copy the model parameters from the selected cluster
                        key_to_set = (number_of_clusters, cluster_num)
                        key_to_copy = (number_of_clusters, cluster_selected)
                        computed_covariance[key_to_set] = computed_covariance[
                            key_to_copy
                        ]
                        cluster_mean_stacked_info[key_to_set] = complete_D_train[
                            point_to_move, :
                        ]
                        cluster_mean_info[key_to_set] = complete_D_train[
                            point_to_move, (window_size - 1) * time_series_col_size :
                        ]

        for cluster_num in range(number_of_clusters):
            print(
                f"length of cluster #{cluster_num} ----> {np.sum(clustered_points == cluster_num)}"
            )

        write_plot(
            clustered_points,
            str_NULL,
            training_indices,
            number_of_clusters,
            write_out_file,
            lambda_parameter,
            beta,
        )

        if old_clustered_points is not None and np.array_equal(
            old_clustered_points, clustered_points
        ):
            print("\n\nCONVERGED!!! BREAKING EARLY!!!")
            break
        old_clustered_points = clustered_points.copy()

    if pool is not None:
        pool.close()
        pool.join()

    # Final calculations remain the same
    train_confusion_matrix_EM = compute_confusion_matrix(
        number_of_clusters, clustered_points, training_indices
    )
    train_confusion_matrix_GMM = compute_confusion_matrix(
        number_of_clusters, gmm_clustered_pts, training_indices
    )
    train_confusion_matrix_kmeans = compute_confusion_matrix(
        number_of_clusters, kmeans_clustered_pts, training_indices
    )
    matching_EM, matching_GMM, matching_Kmeans = compute_matches(
        train_confusion_matrix_EM,
        train_confusion_matrix_GMM,
        train_confusion_matrix_kmeans,
        number_of_clusters,
    )

    compute_f_score(
        matching_EM,
        matching_GMM,
        matching_Kmeans,
        train_confusion_matrix_EM,
        train_confusion_matrix_GMM,
        train_confusion_matrix_kmeans,
        number_of_clusters,
    )

    if compute_BIC:
        bic = computeBIC(
            number_of_clusters,
            time_series_rows_size,
            clustered_points,
            train_cluster_inverse,
            empirical_covariances,
        )
        return clustered_points, train_cluster_inverse, bic

    return clustered_points, train_cluster_inverse
