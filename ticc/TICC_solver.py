"""
This file contains a functional implementation of the TICC solver.
"""

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
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd

from ticc.admm_solver import admm_solver, upper2Full
from ticc.TICC_helper import (
    compute_confusion_matrix,
    computeBIC,
    find_matching,
    getTrainTestSplit,
    updateClusters,
)

import jpviz


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

    # Convert back to JAX arrays and create status strings
    # Keep as JAX arrays since we're inside a JIT-compiled function
    # Note: We don't return status_list in JAX implementation since it's not used
    solutions_list = solutions  # Keep as JAX array

    return solutions_list, convergence_flags


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


def prepare_initial_jax_state(
    clustered_points,
    complete_D_train,
    number_of_clusters,
    window_size,
    time_series_col_size,
    lambda_parameter,
    biased,
    prng_key,
):
    """
    Create initial JAX state from numpy arrays and parameters.
    """
    # Initialize with dummy values that will be updated in first iteration
    dummy_cov_size = window_size * time_series_col_size

    initial_state = {
        "clustered_points": jnp.array(clustered_points),
        "computed_covariances": jnp.zeros(
            (number_of_clusters, dummy_cov_size, dummy_cov_size)
        ),
        "cluster_means": jnp.zeros((number_of_clusters, 1, time_series_col_size)),
        "cluster_means_stacked": jnp.zeros((number_of_clusters, dummy_cov_size)),
        "train_cluster_inverses": jnp.zeros(
            (number_of_clusters, dummy_cov_size, dummy_cov_size)
        ),
        "prng_key": prng_key,
        "converged": False,
    }

    return initial_state


@jax.jit
def update_clusters_jax(lle_matrix, switch_penalty):
    """
    JAX implementation of the updateClusters function using dynamic programming.
    """
    num_points, num_clusters = lle_matrix.shape

    # Initialize DP table
    dp = jnp.full((num_points, num_clusters), jnp.inf)

    # Base case: first time point
    dp = dp.at[0, :].set(lle_matrix[0, :])

    def update_dp_row(carry, t):
        dp_prev = carry

        # For each cluster at time t
        def compute_cluster_cost(k):
            # Cost from staying in same cluster
            same_cluster_cost = dp_prev[k] + lle_matrix[t, k]

            # Cost from switching from any other cluster
            def switch_cost_from_j(j):
                return jnp.where(
                    j == k,
                    jnp.inf,  # Don't consider switching from same cluster
                    dp_prev[j] + lle_matrix[t, k] + switch_penalty,
                )

            switch_costs = jax.vmap(switch_cost_from_j)(jnp.arange(num_clusters))
            min_switch_cost = jnp.min(switch_costs)

            return jnp.minimum(same_cluster_cost, min_switch_cost)

        dp_current = jax.vmap(compute_cluster_cost)(jnp.arange(num_clusters))
        return dp_current, dp_current

    # Run DP for all time points
    _, dp_all = jax.lax.scan(update_dp_row, dp[0, :], jnp.arange(1, num_points))

    # Reconstruct full DP table
    dp_full = jnp.concatenate([dp[0:1, :], dp_all], axis=0)

    # Backtrack to find optimal path
    path = jnp.zeros(num_points, dtype=jnp.int32)

    # Find best final cluster
    final_cluster = jnp.argmin(dp_full[-1, :])

    def backtrack_step(carry, t_rev):
        current_cluster = carry
        t = num_points - 1 - t_rev

        # Use JAX conditional instead of Python if
        def handle_final_point():
            return current_cluster, current_cluster

        def handle_regular_point():
            # Check if we came from same cluster or switched
            same_cost = dp_full[t, current_cluster] + lle_matrix[t + 1, current_cluster]

            def check_switch_from_j(j):
                switch_cost = (
                    dp_full[t, j] + lle_matrix[t + 1, current_cluster] + switch_penalty
                )
                return jnp.abs(dp_full[t + 1, current_cluster] - switch_cost) < 1e-10, j

            was_switch, prev_clusters = jax.vmap(check_switch_from_j)(
                jnp.arange(num_clusters)
            )

            # If any switch matches, use the first one found
            switched_from = jnp.where(
                jnp.any(was_switch),
                prev_clusters[jnp.argmax(was_switch)],
                current_cluster,
            )

            # Choose based on which cost matches
            prev_cluster = jnp.where(
                jnp.abs(dp_full[t + 1, current_cluster] - same_cost) < 1e-10,
                current_cluster,
                switched_from,
            )

            return prev_cluster, prev_cluster

        return jax.lax.cond(
            t == num_points - 1, handle_final_point, handle_regular_point
        )

    _, path_reversed = jax.lax.scan(
        backtrack_step, final_cluster, jnp.arange(num_points)
    )

    # Reverse path and set final cluster
    path = jnp.concatenate([jnp.flip(path_reversed[1:]), jnp.array([final_cluster])])

    return path


def em_update_step(
    state,
    iteration,
    complete_D_train,
    window_size,
    time_series_col_size,
    lambda_parameter,
    beta,
    biased,
    number_of_clusters,
    cluster_reassignment,
):
    """
    Corrected single EM iteration step for JAX scan.
    """
    clustered_points = state["clustered_points"]
    prng_key = state["prng_key"]
    prng_key, subkey = jax.random.split(prng_key)

    # === M-STEP: Correctly compute statistics in parallel ===

    # 1. Get cluster counts
    cluster_counts = jnp.bincount(clustered_points, length=number_of_clusters)
    safe_counts = jnp.maximum(cluster_counts, 1)

    # 2. Compute cluster means using segment_sum
    cluster_sums = jax.ops.segment_sum(
        complete_D_train, clustered_points, num_segments=number_of_clusters
    )
    cluster_means_stacked = cluster_sums / safe_counts[:, jnp.newaxis]

    start_idx = (window_size - 1) * time_series_col_size
    cluster_means = jax.vmap(
        lambda x: jax.lax.dynamic_slice(x, [start_idx], [time_series_col_size])
    )(cluster_means_stacked)

    # 3. Compute empirical covariances (S) using segment_sum
    # This pattern correctly computes per-cluster covariance in a vectorized way.
    def outer_product(x):
        return jnp.outer(x, x)

    centered_data = complete_D_train - cluster_means_stacked[clustered_points]
    sum_of_outer_products = jax.ops.segment_sum(
        jax.vmap(outer_product)(centered_data),
        clustered_points,
        num_segments=number_of_clusters,
    )

    # Denominator for covariance
    cov_denom = jnp.maximum(cluster_counts - (1 - biased), 1)[
        :, jnp.newaxis, jnp.newaxis
    ]
    S_batch = sum_of_outer_products / cov_denom

    # 4. Solve ADMM for all clusters in parallel
    solutions, _ = solve_all_clusters_vmap(list(S_batch), lambda_parameter, window_size)
    solutions_stacked = jnp.stack(solutions)

    # 5. Process ADMM solutions and update state
    S_est_batch = jax.vmap(upper2Full)(solutions_stacked)
    cov_out_batch = jax.vmap(jnp.linalg.inv)(S_est_batch)

    # Update state with new JAX arrays
    state = state.copy()  # Make state mutable for updates
    state["computed_covariances"] = cov_out_batch
    state["train_cluster_inverses"] = S_est_batch
    state["cluster_means"] = cluster_means[:, jnp.newaxis, :]
    state["cluster_means_stacked"] = cluster_means_stacked

    # === E-STEP: Update cluster assignments ===

    cluster_means_stacked_smooth, inv_cov_matrices, log_det_values = (
        prepare_smoothen_data_for_jax_from_state(
            state, number_of_clusters, window_size + 1, time_series_col_size
        )
    )

    lle_all_points_clusters = smoothen_clusters_jax(
        complete_D_train,
        cluster_means_stacked_smooth,
        inv_cov_matrices,
        log_det_values,
        window_size + 1,
        time_series_col_size,
    )

    new_clustered_points = update_clusters_jax(lle_all_points_clusters, beta)

    # Check convergence
    converged = jnp.array_equal(clustered_points, new_clustered_points)

    # === Empty Cluster Reassignment (Corrected Logic) ===

    new_cluster_counts = jnp.bincount(new_clustered_points, length=number_of_clusters)
    has_empty_clusters = jnp.any(new_cluster_counts == 0)

    def reassign_empty_clusters():
        reassign_key, _ = jax.random.split(subkey)
        largest_cluster_id = jnp.argmax(new_cluster_counts)

        # Get indices of points in the largest cluster, padded with -1 to keep a static shape.
        largest_cluster_indices = jnp.where(
            new_clustered_points == largest_cluster_id,
            size=len(new_clustered_points),
            fill_value=-1,
        )[0]

        # 1. Create a boolean mask of valid indices (where index != -1).
        is_valid_index = largest_cluster_indices != -1

        # 2. Convert the mask to a float-based probability vector (weights).
        weights = is_valid_index.astype(jnp.float32)
        # Normalize the weights so they sum to 1.
        weights /= jnp.maximum(jnp.sum(weights), 1e-10)

        # 3. Use the 'p' argument to sample from the full array using the weights.
        # This will only select from indices where the weight is non-zero.
        start_point_idx = jax.random.choice(
            reassign_key, largest_cluster_indices, p=weights
        )

        reassign_indices = start_point_idx + jnp.arange(cluster_reassignment)
        reassign_indices = jnp.mod(reassign_indices, len(new_clustered_points))

        first_empty_cluster = jnp.argmin(new_cluster_counts)

        return new_clustered_points.at[reassign_indices].set(first_empty_cluster)

    final_clustered_points = jax.lax.cond(
        has_empty_clusters & (iteration > 0),
        reassign_empty_clusters,
        lambda: new_clustered_points,
    )

    # Update state for next iteration
    new_state = {
        "clustered_points": final_clustered_points,
        "computed_covariances": state["computed_covariances"],
        "cluster_means": state["cluster_means"],
        "cluster_means_stacked": state["cluster_means_stacked"],
        "train_cluster_inverses": state["train_cluster_inverses"],
        "prng_key": prng_key,
        "converged": converged,
    }

    return new_state, {"iteration": iteration, "converged": converged}


def prepare_smoothen_data_for_jax_from_state(state, number_of_clusters, num_blocks, n):
    """
    Extract smoothen data from JAX state instead of dictionaries.
    """
    sub_features = (num_blocks - 1) * n

    cluster_means_stacked = state["cluster_means_stacked"]
    computed_covariances = state["computed_covariances"]

    # Extract sub-matrices and compute inverses and log-dets
    cov_sub_matrices = computed_covariances[:, :sub_features, :sub_features]
    inv_cov_matrices = jax.vmap(jnp.linalg.inv)(cov_sub_matrices)
    log_det_values = jax.vmap(lambda x: jnp.log(jnp.linalg.det(x)))(cov_sub_matrices)

    return cluster_means_stacked, inv_cov_matrices, log_det_values


@partial(
    jax.jit,
    static_argnames=(
        "window_size",
        "time_series_col_size",
        "lambda_parameter",
        "beta",
        "maxIters",
        "cluster_reassignment",
        "biased",
        "number_of_clusters",
    ),
)
def ticc_jax_fit(
    initial_state,
    complete_D_train,
    window_size: int,
    time_series_col_size: int,
    lambda_parameter: float,
    beta: float,
    maxIters: int,
    cluster_reassignment: int,
    biased: bool,
    number_of_clusters: int,
):
    """
    JAX-compiled main TICC fitting loop using scan.
    """

    def scan_em_step(state, iteration):
        return em_update_step(
            state,
            iteration,
            complete_D_train,
            window_size,
            time_series_col_size,
            lambda_parameter,
            beta,
            biased,
            number_of_clusters,
            cluster_reassignment,
        )

    final_state, history = jax.lax.scan(
        scan_em_step, initial_state, jnp.arange(maxIters)
    )

    return final_state, history


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
    write_out_file=False,
    prefix_string="",
    compute_BIC=False,
    cluster_reassignment=20,
    biased=False,
    seed=None,
    jax_profile=False,
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
    prepare_out_directory(prefix_string, lambda_parameter, number_of_clusters)

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

    # Use JAX-compiled implementation
    print("Using JAX-compiled TICC implementation...")

    # Prepare initial state
    if seed is None:
        seed = np.random.randint(0, 1 << 31)
    prng_key = jax.random.PRNGKey(seed)
    initial_state = prepare_initial_jax_state(
        clustered_points,
        complete_D_train,
        number_of_clusters,
        window_size,
        time_series_col_size,
        lambda_parameter,
        biased,
        prng_key,
    )

    # Run JAX-compiled fit
    start = time.time()
    if jax_profile:
        dot_graph = jpviz.draw(ticc_jax_fit, static_argnums=list(range(2, 10)))(
            initial_state,
            jnp.array(complete_D_train),
            window_size,
            time_series_col_size,
            lambda_parameter,
            beta,
            maxIters,
            cluster_reassignment,
            biased,
            number_of_clusters,
        )
        dot_graph.write(f"{prefix_string}jax.dot")
        with open(f"{prefix_string}jax.svg", "wb") as f:
            f.write(dot_graph.create_svg())
        dot_graph.write_png(f"{prefix_string}jax.png")

    final_state, history = ticc_jax_fit(
        initial_state,
        jnp.array(complete_D_train),
        window_size,
        time_series_col_size,
        lambda_parameter,
        beta,
        maxIters,
        cluster_reassignment,
        biased,
        number_of_clusters,
    )
    final_state["clustered_points"].block_until_ready()

    # Extract results from final state
    clustered_points = np.array(final_state["clustered_points"])

    end = time.time()
    print(f"JAX TICC fitting took {end - start:.4f} seconds")

    # Convert JAX state back to dictionary format for compatibility
    train_cluster_inverse = {}
    computed_covariance = {}
    cluster_mean_info = {}
    cluster_mean_stacked_info = {}
    empirical_covariances = {}

    for cluster in range(number_of_clusters):
        # Convert arrays back to the expected dictionary format
        computed_covariance[(number_of_clusters, cluster)] = np.array(
            final_state["computed_covariances"][cluster]
        )
        train_cluster_inverse[cluster] = np.array(
            final_state["train_cluster_inverses"][cluster]
        )
        cluster_mean_info[(number_of_clusters, cluster)] = np.array(
            final_state["cluster_means"][cluster]
        )
        cluster_mean_stacked_info[(number_of_clusters, cluster)] = np.array(
            final_state["cluster_means_stacked"][cluster]
        )
        # Empirical covariances not stored in state, set to dummy values
        empirical_covariances[cluster] = computed_covariance[
            (number_of_clusters, cluster)
        ]

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
