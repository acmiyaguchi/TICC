import jax.numpy as jnp
import jax
import math


def _ij2symmetric(i, j, size):
    return (size * (size + 1)) / 2 - (size - i) * (size - i + 1) / 2 + j - i


@jax.jit
def upper2Full(a):
    """Convert upper triangular matrix to full matrix"""
    # FIX: Use math.sqrt to calculate 'n' as a concrete Python integer
    # This avoids creating a JAX tracer for a shape definition.
    n = int((-1 + math.sqrt(1 + 8 * a.shape[0])) / 2)
    A = jnp.zeros([n, n])
    triu_indices = jnp.triu_indices(n)
    A = A.at[triu_indices].set(a)
    temp = jnp.diag(A)
    A = (A + A.T) - jnp.diag(temp)
    return A


@jax.jit
def Prox_logdet(S, A, eta):
    """Proximal operator for log determinant"""
    d, q = jnp.linalg.eigh(eta * A - S)
    # Match the NumPy matrix behavior more closely
    X_var = (
        (1 / (2 * eta))  # Remove float() conversion to avoid tracer issues
        * q
        @ jnp.diag(d + jnp.sqrt(jnp.square(d) + (4 * eta) * jnp.ones(d.shape)))
        @ q.T
    )
    x_var = X_var[
        jnp.triu_indices(S.shape[1])
    ]  # extract upper triangular part as update variable
    return x_var.reshape(-1, 1)


@jax.jit
def ADMM_x(z, u, S, rho):
    """ADMM update for x variable"""
    a = z - u
    A = upper2Full(a)
    eta = rho
    x_update = Prox_logdet(S, A, eta)
    return x_update.T.reshape(-1)


def ADMM_z(x, u, lamb, rho, numBlocks, sizeBlocks, length):
    """ADMM update for z variable
    
    Note: This function contains Python loops and cannot be JIT compiled.
    This will be optimized in a later phase with vectorized operations.
    """
    a = x + u
    probSize = numBlocks * sizeBlocks
    z_update = jnp.zeros(length)

    # Follow the exact same logic as NumPy version
    for i in range(numBlocks):
        elems = numBlocks if i == 0 else (2 * numBlocks - 2 * i) / 2  # i=0 is diagonal
        for j in range(sizeBlocks):
            startPoint = j if i == 0 else 0
            for k in range(startPoint, sizeBlocks):
                # Create location lists exactly as in NumPy version
                locList_0 = jnp.array(
                    [(L + i) * sizeBlocks + j for L in range(int(elems))]
                )
                locList_1 = jnp.array([L * sizeBlocks + k for L in range(int(elems))])

                if i == 0:
                    # Calculate lambda sum
                    lamSum = 0
                    for idx in range(len(locList_0)):
                        lamSum += lamb[locList_0[idx], locList_1[idx]]

                    # Calculate indices
                    indices = jnp.zeros(len(locList_0))
                    for idx in range(len(locList_0)):
                        indices = indices.at[idx].set(
                            _ij2symmetric(locList_0[idx], locList_1[idx], probSize)
                        )
                else:
                    # Calculate lambda sum
                    lamSum = 0
                    for idx in range(len(locList_0)):
                        lamSum += lamb[locList_1[idx], locList_0[idx]]

                    # Calculate indices
                    indices = jnp.zeros(len(locList_0))
                    for idx in range(len(locList_0)):
                        indices = indices.at[idx].set(
                            _ij2symmetric(locList_1[idx], locList_0[idx], probSize)
                        )

                # Calculate point sum
                pointSum = 0
                for index in indices:
                    pointSum += a[int(index)]
                rhoPointSum = rho * pointSum

                # Calculate soft threshold (exactly as in NumPy version)
                ans = 0
                # If answer is positive
                if rhoPointSum > lamSum:
                    ans = max((rhoPointSum - lamSum) / (rho * elems), 0)
                elif rhoPointSum < -1 * lamSum:
                    ans = min((rhoPointSum + lamSum) / (rho * elems), 0)

                # Update z for all indices
                for index in indices:
                    z_update = z_update.at[int(index)].set(ans)

    return z_update


@jax.jit
def ADMM_u(u, x, z):
    """ADMM update for u variable"""
    return u + x - z


def check_convergence(x, z, z_old, u, rho, length, e_abs, e_rel, verbose=False):
    """
    Check convergence criteria for ADMM
    
    Note: Cannot be JIT compiled due to Python print statements and control flow

    Returns (boolean shouldStop, primal residual value, primal threshold,
             dual residual value, dual threshold)
    """
    norm = jnp.linalg.norm
    r = x - z
    s = rho * (z - z_old)
    # Primal and dual thresholds. Add .0001 to prevent the case of 0.
    e_pri = math.sqrt(length) * e_abs + e_rel * max(norm(x), norm(z)) + 0.0001
    e_dual = math.sqrt(length) * e_abs + e_rel * norm(rho * u) + 0.0001
    # Primal and dual residuals
    res_pri = norm(r)
    res_dual = norm(s)
    if verbose:
        # Debugging information to print convergence criteria values
        print("  r:", res_pri)
        print("  e_pri:", e_pri)
        print("  s:", res_dual)
        print("  e_dual:", e_dual)
    stop = (res_pri <= e_pri) and (res_dual <= e_dual)
    return (stop, res_pri, e_pri, res_dual, e_dual)


def admm_solver(
    S,
    lamb,
    num_stacked,
    size_blocks,
    rho,
    maxIters=1000,
    eps_abs=1e-6,
    eps_rel=1e-6,
    verbose=False,
    rho_update_func=None,
):
    """
    Solve Sparse Inverse Covariance Selection problem via ADMM
    
    Note: The main solver loop is not JIT compiled due to Python control flow
    and the ADMM_z function containing Python loops. Individual update functions
    are JIT compiled for performance.

    Parameters:
    -----------
    S : jax.numpy.ndarray
        Empirical covariance matrix
    lamb : float
        Regularization parameter
    num_stacked : int
        Number of blocks
    size_blocks : int
        Size of each block
    rho : float
        ADMM parameter
    maxIters : int, optional
        Maximum number of iterations
    eps_abs : float, optional
        Absolute tolerance
    eps_rel : float, optional
        Relative tolerance
    verbose : bool, optional
        Whether to print debug information
    rho_update_func : callable, optional
        Function to update rho

    Returns:
    --------
    x : jax.numpy.ndarray
        Solution to the problem
    status : str
        Status of the solver
    """
    probSize = num_stacked * size_blocks
    length = int(probSize * (probSize + 1) / 2)

    # Initialize variables
    x = jnp.zeros(length)
    z = jnp.zeros(length)
    u = jnp.zeros(length)

    status = "Incomplete: max iterations reached"

    # ADMM iterations
    for i in range(maxIters):
        z_old = jnp.copy(z)

        # Update x
        x = ADMM_x(z, u, S, rho)

        # Update z
        z = ADMM_z(x, u, lamb, rho, num_stacked, size_blocks, length)

        # Update u
        u = ADMM_u(u, x, z)

        if i != 0:
            # Check convergence
            stop, res_pri, e_pri, res_dual, e_dual = check_convergence(
                x, z, z_old, u, rho, length, eps_abs, eps_rel, verbose
            )

            if stop:
                status = "Optimal"
                break

            # Update rho if needed
            curr_rho = rho
            if rho_update_func:
                rho = rho_update_func(rho, res_pri, e_pri, res_dual, e_dual)
                if rho != curr_rho:
                    scale = curr_rho / rho
                    u = scale * u

        if verbose:
            print("Iteration %d" % i)

    return x, status
