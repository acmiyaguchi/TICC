"""
This file contains a functional implementation of the TICC solver.
"""

import collections
import errno
import os
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import faiss
import gmmx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import tqdm

from ticc.jax_admm_solver import admm_solver
from ticc.TICC_helper import (
    compute_confusion_matrix,
    computeBIC,
    find_matching,
    getTrainTestSplit,
    updateClusters,
    upperToFull,
)


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

    # Create vectorized version of the JAX ADMM solver JIT function
    batch_solver = jax.vmap(
        admm_solver,
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
    status_list = [
        "Optimal" if bool(flag) else "Incomplete: max iterations reached"
        for flag in convergence_flags
    ]

    return solutions_list, status_list


@jax.jit
def calculate_lle(point_data, cluster_mean, inv_cov, log_det):
    """
    Calculate Log-Likelihood Error for a single data point against single cluster parameters.

    Args:
        point_data: Data point (vector)
        cluster_mean: Cluster mean (vector)
        inv_cov: Inverse covariance matrix
        log_det: Log determinant of covariance matrix

    Returns:
        lle: Log-likelihood error (scalar)
    """
    x = point_data - cluster_mean
    # Compute x^T * inv_cov * x + log_det
    lle = jnp.dot(x, jnp.dot(inv_cov, x)) + log_det
    return lle


def smoothen_clusters_jax(
    complete_D_train,
    cluster_means_stacked,
    inv_cov_matrices,
    log_det_values,
    num_blocks,
    n,
):
    """
    Memory-efficient JAX implementation of smoothen_clusters.
    Uses lax.map for sequential processing of points and vmap for parallel processing of clusters.
    """
    clustered_points_len = complete_D_train.shape[0]
    relevant_features = (num_blocks - 1) * n
    relevant_means = cluster_means_stacked[:, :relevant_features]

    # This inner function, vectorized over clusters, is perfect as-is.
    # It calculates all LLEs for a single point in parallel.
    lle_for_one_point = jax.vmap(calculate_lle, in_axes=(None, 0, 0, 0))

    def compute_lles_for_point(point_idx):
        # This wrapper handles the logic for a single point
        valid = point_idx + num_blocks - 2 < clustered_points_len

        def compute_lles():
            point_data = complete_D_train[point_idx, :relevant_features]
            return lle_for_one_point(
                point_data, relevant_means, inv_cov_matrices, log_det_values
            )

        return jax.lax.cond(
            valid, compute_lles, lambda: jnp.zeros(cluster_means_stacked.shape[0])
        )

    # FIX: Use jax.lax.map to iterate over points sequentially, saving memory.
    LLE_all_points_clusters = jax.lax.map(
        compute_lles_for_point, jnp.arange(clustered_points_len)
    )

    return LLE_all_points_clusters


def prepare_smoothen_data_for_jax(
    cluster_mean_stacked_info,
    computed_covariance,
    number_of_clusters,
    num_blocks,
    n,
):
    """
    Convert dictionary-based cluster data to stacked JAX arrays for vectorized processing.

    Args:
        cluster_mean_stacked_info: Dictionary of cluster means
        computed_covariance: Dictionary of covariance matrices
        number_of_clusters: Number of clusters
        num_blocks: Number of blocks
        n: Feature dimension

    Returns:
        cluster_means_stacked: JAX array (num_clusters, num_features)
        inv_cov_matrices: JAX array (num_clusters, sub_features, sub_features)
        log_det_values: JAX array (num_clusters,)
    """
    # Extract relevant sub-matrix dimension
    sub_features = (num_blocks - 1) * n

    # Pre-allocate arrays
    cluster_means_list = []
    inv_cov_list = []
    log_det_list = []

    for cluster in range(number_of_clusters):
        # Get cluster mean
        cluster_mean = cluster_mean_stacked_info[(number_of_clusters, cluster)]
        cluster_means_list.append(cluster_mean)

        # Get covariance sub-matrix and compute inverse and log-det
        cov_matrix = computed_covariance[(number_of_clusters, cluster)][
            0:sub_features, 0:sub_features
        ]
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        log_det_cov = np.log(np.linalg.det(cov_matrix))

        inv_cov_list.append(inv_cov_matrix)
        log_det_list.append(log_det_cov)

    # Convert to JAX arrays
    cluster_means_stacked = jnp.array(cluster_means_list)
    inv_cov_matrices = jnp.array(inv_cov_list)
    log_det_values = jnp.array(log_det_list)

    return cluster_means_stacked, inv_cov_matrices, log_det_values


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

    # Use JAX vectorized smoothen_clusters for better performance
    cluster_means_stacked, inv_cov_matrices, log_det_values = (
        prepare_smoothen_data_for_jax(
            trained_model["cluster_mean_stacked_info"],
            trained_model["computed_covariance"],
            trained_model["number_of_clusters"],
            num_blocks,
            trained_model["time_series_col_size"],
        )
    )

    lle_all_points_clusters = smoothen_clusters_jax(
        jnp.array(test_data),
        cluster_means_stacked,
        inv_cov_matrices,
        log_det_values,
        num_blocks,
        trained_model["time_series_col_size"],
    )

    # Convert back to numpy for compatibility with updateClusters
    lle_all_points_clusters = np.array(lle_all_points_clusters)

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

    for iters in tqdm.tqdm(range(maxIters)):
        train_clusters_arr = collections.defaultdict(list)
        for point, cluster_num in enumerate(clustered_points):
            train_clusters_arr[cluster_num].append(point)

        len_train_clusters = {
            k: len(train_clusters_arr[k]) for k in range(number_of_clusters)
        }

        # JAX solver logic moved directly into fit loop
        # Create empty list for cluster data
        cluster_data_list = []
        cluster_indices = []  # Track which clusters have data

        # Loop to prepare D_train arrays for each cluster
        for cluster in range(number_of_clusters):
            cluster_length = len_train_clusters[cluster]
            if cluster_length != 0:
                indices = train_clusters_arr[cluster]
                D_train = np.zeros([cluster_length, window_size * time_series_col_size])
                for i in range(cluster_length):
                    point = indices[i]
                    D_train[i, :] = complete_D_train[point, :]

                # Store cluster means
                cluster_mean_info[number_of_clusters, cluster] = np.mean(
                    D_train, axis=0
                )[
                    (window_size - 1) * time_series_col_size : window_size
                    * time_series_col_size
                ].reshape([1, time_series_col_size])
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
