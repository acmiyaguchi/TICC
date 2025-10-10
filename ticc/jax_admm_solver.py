import jax.numpy as jnp
import jax
import math
from functools import partial  # Import partial


def _ij2symmetric(i, j, size):
    return (size * (size + 1)) // 2 - (size - i) * (size - i + 1) // 2 + j - i


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


@partial(jax.jit, static_argnames=("numBlocks", "sizeBlocks", "length"))
def ADMM_z(x, u, lamb, rho, numBlocks, sizeBlocks, length):
    a = x + u
    probSize = numBlocks * sizeBlocks
    max_elems = numBlocks

    def process_i_block(z_carry, i):
        elems = jnp.where(i == 0, numBlocks, (2 * numBlocks - 2 * i) / 2.0)

        def process_j_block(z_inner, j):
            def process_k_value(z_k, k):
                valid = jnp.where(i == 0, k >= j, True)

                def do_computation():
                    L_range = jnp.arange(max_elems)
                    valid_L = L_range < elems.astype(jnp.int32)

                    locList_0 = (L_range + i) * sizeBlocks + j
                    locList_1 = L_range * sizeBlocks + k

                    locA = jnp.where(i == 0, locList_0, locList_1)
                    locB = jnp.where(i == 0, locList_1, locList_0)

                    lamSum = jnp.sum(lamb[locA, locB] * valid_L)

                    indices_raw = jax.vmap(_ij2symmetric, in_axes=(0, 0, None))(
                        locA, locB, probSize
                    )

                    point_contribs = a[indices_raw.astype(jnp.int32)] * valid_L
                    pointSum = jnp.sum(point_contribs)
                    rhoPointSum = rho * pointSum

                    ans = jnp.where(
                        rhoPointSum > lamSum,
                        jnp.maximum((rhoPointSum - lamSum) / (rho * elems), 0),
                        jnp.where(
                            rhoPointSum < -lamSum,
                            jnp.minimum((rhoPointSum + lamSum) / (rho * elems), 0),
                            0.0,
                        ),
                    )

                    indices_int = indices_raw.astype(jnp.int32)
                    original_values = z_k[indices_int]
                    update_values = jnp.where(valid_L, ans, original_values)
                    z_updated = z_k.at[indices_int].set(update_values)

                    return z_updated

                updated_z = jax.lax.cond(valid, do_computation, lambda: z_k)
                return updated_z, None

            # This part of the scan structure is perfect as-is
            z_after_k, _ = jax.lax.scan(
                process_k_value, z_inner, jnp.arange(sizeBlocks)
            )
            return z_after_k, None

        z_after_j, _ = jax.lax.scan(process_j_block, z_carry, jnp.arange(sizeBlocks))
        return z_after_j, None

    z_final, _ = jax.lax.scan(process_i_block, jnp.zeros(length), jnp.arange(numBlocks))
    return z_final


@jax.jit
def ADMM_u(u, x, z):
    """ADMM update for u variable"""
    return u + x - z


@jax.jit
def check_convergence_jit(x, z, z_old, u, rho, length, e_abs, e_rel):
    """
    JIT-compiled convergence check without print statements

    Returns (boolean shouldStop, primal residual value, primal threshold,
             dual residual value, dual threshold)
    """
    norm = jnp.linalg.norm
    r = x - z
    s = rho * (z - z_old)

    # FIX: Use jnp.sqrt instead of math.sqrt to handle tracers
    sqrt_length = jnp.sqrt(length.astype(float))  # Cast length to float for sqrt

    e_pri = sqrt_length * e_abs + e_rel * jnp.maximum(norm(x), norm(z)) + 0.0001
    e_dual = sqrt_length * e_abs + e_rel * norm(rho * u) + 0.0001

    res_pri = norm(r)
    res_dual = norm(s)
    stop = (res_pri <= e_pri) & (res_dual <= e_dual)

    return (stop, res_pri, e_pri, res_dual, e_dual)


@partial(
    jax.jit,
    static_argnames=("num_stacked", "size_blocks", "maxIters", "eps_abs", "eps_rel"),
)
def admm_solver_jit(S, lamb, num_stacked, size_blocks, rho, maxIters, eps_abs, eps_rel):
    """The fully JIT-compiled solver"""
    probSize = num_stacked * size_blocks
    length = int(probSize * (probSize + 1) / 2)

    def admm_step(state, iteration):
        """Single ADMM iteration step with dictionary state for clarity"""

        # This function defines the logic for a single computation step.
        def update_fn(operand_state):
            x_prev, z_prev, u_prev = (
                operand_state["x"],
                operand_state["z"],
                operand_state["u"],
            )

            z_old = z_prev
            x_new = ADMM_x(z_prev, u_prev, S, rho)
            z_new = ADMM_z(x_new, u_prev, lamb, rho, num_stacked, size_blocks, length)
            u_new = ADMM_u(u_prev, x_new, z_new)

            stop, _, _, _, _ = check_convergence_jit(
                x_new, z_new, z_old, u_prev, rho, length, eps_abs, eps_rel
            )
            is_converged = jnp.where(iteration > 0, stop, False)

            return {"x": x_new, "z": z_new, "u": u_new, "done": is_converged}

        # If the previous state was 'done', just pass it through.
        # Otherwise, run the update function.
        new_state = jax.lax.cond(
            state["done"],
            lambda s: s,  # If done, the new state is the same as the old state.
            update_fn,
            state,  # Pass state as the operand to the selected function.
        )

        # The final 'done' flag is a combination of its previous state and the new one.
        final_done_flag = state["done"] | new_state["done"]

        # Always carry forward the final 'done' status.
        new_state["done"] = final_done_flag

        return new_state, final_done_flag

    # Initialize state with dictionary for clarity
    initial_state = {
        "x": jnp.zeros(length),
        "z": jnp.zeros(length),
        "u": jnp.zeros(length),
        "done": False,
    }

    # Run ADMM iterations using scan
    final_state, converged_history = jax.lax.scan(
        admm_step, initial_state, jnp.arange(maxIters)
    )

    any_converged = jnp.any(converged_history)
    return final_state["x"], any_converged


# Wrapper function to maintain the original signature
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

    Now fully JIT-compiled using jax.lax.scan for maximum performance!

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
        Whether to print debug information (disabled in JIT version)
    rho_update_func : callable, optional
        Function to update rho (not supported in JIT version)

    Returns:
    --------
    x : jax.numpy.ndarray
        Solution to the problem
    status : str
        Status of the solver
    """
    # Call the JIT-compiled version
    x_final, any_converged = admm_solver_jit(
        S, lamb, num_stacked, size_blocks, rho, maxIters, eps_abs, eps_rel
    )

    # Convert status object back to a Python string if needed
    status = "Optimal" if any_converged else "Incomplete: max iterations reached"

    # Verbose and rho_update are not supported in the JIT version, handle outside if needed
    if verbose:
        print(f"JIT solver finished with status: {status}")

    return x_final, status
