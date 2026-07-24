"""
Auxiliary functions for trajectory post-processing (NumPy-based).
"""
import numpy as np

# Constants
AU_TO_M = 1.495978707e11


def _cross_np(a, b):
    """3D cross product helper."""
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])


def _resolve_kepler_n_iter(n_iter):
    """Resolve Kepler integration sub-steps, defaulting to optimizer setting."""
    if n_iter is not None:
        return int(n_iter)
    from optimizer.orbit_utils import get_kepler_substeps
    return int(get_kepler_substeps())


def _dL_dt_np(L, p, f, g, h, k, mu):
    """
    Compute dL/dt for Keplerian motion in MEE formulation (NumPy version).
    
    Under pure Keplerian motion, the first 5 MEE elements are constant.
    Only true longitude L changes according to:
    
    dL/dt = sqrt(μ*p) * (w/p)²
    
    where w = 1 + f*cos(L) + g*sin(L)
    
    Args:
        L: Current true longitude
        p, f, g, h, k: MEE elements (p, f, g constant; h, k not used)
        mu: Gravitational parameter
    
    Returns:
        dL/dt: Rate of change of true longitude
    """
    cosL = np.cos(L)
    sinL = np.sin(L)
    w = 1.0 + f*cosL + g*sinL
    
    # dL/dt = sqrt(μ*p) * (w/p)²
    dLdt = np.sqrt(mu * p) * (w / p)**2

    return dLdt


def kepler_coast_np(x, dt, mu, n_iter=25):
    """
    Propagate MEE state under Keplerian motion using RK4 integration (NumPy version).
    
    Key insight: Under pure Keplerian motion, the first 5 MEE elements
    (p, f, g, h, k) are constant. Only L (true longitude) changes.
    This is much more efficient than iterative Kepler equation solving.
    
    Uses Runge-Kutta 4th order to integrate: dL/dt = n = sqrt(mu/a³)
    
    Args:
        x: 6-element array [p, f, g, h, k, L]
        dt: Time step
        mu: Gravitational parameter
        n_iter: Number of RK4 sub-steps for accuracy
    
    Returns:
        6-element array with propagated state [p, f, g, h, k, L_new]
    """
    p, f, g, h, k, L = x
    
    # RK4 integration of L with n_iter sub-steps
    dt_sub = dt / n_iter
    L_current = L
    
    for _ in range(n_iter):
        # RK4 coefficients
        k1 = _dL_dt_np(L_current,                  p, f, g, h, k, mu)
        k2 = _dL_dt_np(L_current + dt_sub/2 * k1,  p, f, g, h, k, mu)
        k3 = _dL_dt_np(L_current + dt_sub/2 * k2,  p, f, g, h, k, mu)
        k4 = _dL_dt_np(L_current + dt_sub * k3,    p, f, g, h, k, mu)
        
        # Update L
        L_current = L_current + (dt_sub / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return np.array([p, f, g, h, k, L_current])


def apply_dv_np(x, dv, mu):
    """
    Apply delta-v impulse to MEE state (NumPy version).
    
    Args:
        x: 6-element array [p, f, g, h, k, L]
        dv: 3-element array [dvR, dvT, dvN] in RTN frame
        mu: Gravitational parameter
    
    Returns:
        6-element array with updated state
    """
    r, v = mee2rv(x, mu)

    rhat = r / np.linalg.norm(r)
    hhat = _cross_np(r, v)
    hhat = hhat / np.linalg.norm(hhat)
    that = _cross_np(hhat, rhat)

    dv_inertial = dv[0] * rhat + dv[1] * that + dv[2] * hhat
    v_post = v + dv_inertial
    return rv2mee(r, v_post, mu)


def mee2rv(state, mu):
    """Convert MEE state [p,f,g,h,k,L] to inertial Cartesian (r,v)."""
    p, f, g, h, k, L = state
    cosL, sinL = np.cos(L), np.sin(L)
    w = 1.0 + f * cosL + g * sinL
    s2 = 1.0 + h * h + k * k

    fhat = np.array([(1.0 - k * k + h * h) / s2, (2.0 * h * k) / s2, (-2.0 * k) / s2])
    ghat = np.array([(2.0 * h * k) / s2, (1.0 + k * k - h * h) / s2, (2.0 * h) / s2])

    r = (p / w) * (cosL * fhat + sinL * ghat)
    v = np.sqrt(mu / p) * (-(g + sinL) * fhat + (f + cosL) * ghat)
    return r, v


def rv2mee(r, v, mu):
    """Convert inertial Cartesian (r,v) to MEE state [p,f,g,h,k,L]."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    r_norm = np.linalg.norm(r)
    h_vec = _cross_np(r, v)
    h_norm = np.linalg.norm(h_vec)

    p = h_norm**2 / mu
    hhat = h_vec / h_norm
    den = 1.0 + hhat[2]
    h = -hhat[1] / den
    k = hhat[0] / den
    s2 = 1.0 + h * h + k * k

    fhat = np.array([(1.0 - k * k + h * h) / s2, (2.0 * h * k) / s2, (-2.0 * k) / s2])
    ghat = np.array([(2.0 * h * k) / s2, (1.0 + k * k - h * h) / s2, (2.0 * h) / s2])

    x_plane = np.dot(r, fhat)
    y_plane = np.dot(r, ghat)
    L = np.arctan2(y_plane, x_plane)

    e_vec = _cross_np(v, h_vec) / mu - r / r_norm
    f = np.dot(e_vec, fhat)
    g = np.dot(e_vec, ghat)

    return np.array([p, f, g, h, k, L])


def mee2cart(state):
    """
    Convert MEE state to Cartesian coordinates.
    
    Args:
        state: 6-element array [p, f, g, h, k, L]
    
    Returns:
        3-element array [x, y, z] in Cartesian coordinates
    """
    r, _ = mee2rv(state, mu=1.0)
    return r


def reconstruct_trajectory(
    mee0, DV_op, tau_op, T_opt, mu, n_sub=100, return_final_state=False, n_iter=None
):
    """
    Reconstruct full spacecraft trajectory from optimized solution.
    
    Args:
        mee0: Initial MEE state
        DV_op: Delta-v matrix (3 x N_IMP)
        tau_op: Normalized impulse times
        T_opt: Optimal transfer time
        mu: Gravitational parameter
        n_sub: Points per segment
        n_iter: RK sub-steps used in each coast propagation
    
    Returns:
        By default: list of trajectory segments, each segment is (n_sub x 6) array
        If return_final_state=True: (segments, x_final_post_impulse)
    """
    N_IMP = DV_op.shape[1]
    n_iter = _resolve_kepler_n_iter(n_iter)
    segments = []
    x_now = mee0.copy()
    t_now = 0.0

    for i in range(N_IMP):
        t_imp = tau_op[i] * T_opt
        dt = t_imp - t_now

        # Coast segment
        times = np.linspace(0, dt, n_sub)
        seg = np.zeros((n_sub, 6))
        for j, step_t in enumerate(times):
            seg[j] = kepler_coast_np(x_now, step_t, mu, n_iter=n_iter)
        segments.append(seg)

        # Apply impulse at end of segment
        x_now = apply_dv_np(seg[-1], DV_op[:, i], mu)
        t_now = t_imp

    # Final coast
    if t_now < T_opt:
        dt_final = T_opt - t_now
        times = np.linspace(0, dt_final, n_sub)
        seg = np.zeros((n_sub, 6))
        for j, step_t in enumerate(times):
            seg[j] = kepler_coast_np(x_now, step_t, mu, n_iter=n_iter)
        segments.append(seg)

    if return_final_state:
        return segments, x_now

    return segments


def compute_arrival_error(
    segments, mee_target, L0_target, T_opt, mu, final_state_mee=None, n_iter=None
):
    """
    Compute spacecraft arrival error relative to target.
    
    Args:
        segments: List of trajectory segments
        mee_target: Target MEE state [p, f, g, h, k] (without L)
        L0_target: Target initial true longitude
        T_opt: Transfer time
        mu: Gravitational parameter
    
    Args:
        final_state_mee: Optional final MEE state to use for arrival (post-impulse)
        n_iter: RK sub-steps used for target coast propagation

    Returns:
        Dictionary with error metrics
    """
    n_iter = _resolve_kepler_n_iter(n_iter)

    # Target final state
    target_mee_f = kepler_coast_np(
        np.concatenate([[mee_target[0]], mee_target[1:], [L0_target]]),
        T_opt, mu, n_iter=n_iter
    )
    r_target = mee2cart(target_mee_f)

    # Spacecraft final state (prefer explicit post-impulse state when provided)
    if final_state_mee is None:
        final_state_mee = segments[-1][-1]
    r_arrival = mee2cart(final_state_mee)

    pos_error_vec = r_arrival - r_target
    pos_error = np.linalg.norm(pos_error_vec)

    return {
        'r_target': r_target,
        'r_arrival': r_arrival,
        'pos_error_au': pos_error,
        'pos_error_m': pos_error * AU_TO_M,
        'pos_error_vec': pos_error_vec
    }


def compute_model_parity(mee0, DV_op, tau_op, T_opt, mu):
    """
    Compare final states from NumPy post-processing and optimizer symbolic model.

    Args:
        mee0: Initial MEE state [p, f, g, h, k, L]
        DV_op: Delta-v matrix (3 x N_IMP)
        tau_op: Normalized impulse times
        T_opt: Optimal transfer time
        mu: Gravitational parameter
    Returns:
        Dictionary with final states and mismatch metrics.
    """
    import casadi as ca
    from optimizer.orbit_utils import kepler_coast_sym, apply_dv_sym, get_kepler_substeps

    n_kepler = get_kepler_substeps()

    # NumPy reconstruction with exactly the same segment chronology as the optimizer.
    x_np = np.array(mee0, dtype=float).copy()
    t_prev = 0.0
    for i in range(DV_op.shape[1]):
        t_i = float(tau_op[i] * T_opt)
        dt = t_i - t_prev
        x_np = kepler_coast_np(x_np, dt, mu, n_iter=n_kepler)
        x_np = apply_dv_np(x_np, DV_op[:, i], mu)
        t_prev = t_i
    x_np = kepler_coast_np(x_np, float(T_opt - t_prev), mu, n_iter=n_kepler)

    # Symbolic-model reconstruction (same functions used by NLP).
    x_sym = ca.DM(np.array(mee0, dtype=float).reshape(6, 1))
    t_prev = 0.0
    for i in range(DV_op.shape[1]):
        t_i = float(tau_op[i] * T_opt)
        dt = t_i - t_prev
        x_sym = kepler_coast_sym(x_sym, dt, mu)
        x_sym = apply_dv_sym(x_sym, ca.DM(DV_op[:, i]), mu)
        t_prev = t_i
    x_sym = kepler_coast_sym(x_sym, float(T_opt - t_prev), mu)
    x_sym_np = np.array(x_sym.full()).reshape(-1)

    mee_diff = x_np - x_sym_np
    dL_wrap = np.arctan2(np.sin(mee_diff[5]), np.cos(mee_diff[5]))
    mee_abs = np.abs(mee_diff)
    mee_abs_wrapped = mee_abs.copy()
    mee_abs_wrapped[5] = np.abs(dL_wrap)

    r_np = mee2cart(x_np)
    r_sym = mee2cart(x_sym_np)
    r_diff = r_np - r_sym

    return {
        'x_final_np': x_np,
        'x_final_sym': x_sym_np,
        'mee_diff_raw': mee_diff,
        'dL_wrapped': dL_wrap,
        'mee_abs_wrapped': mee_abs_wrapped,
        'r_final_np': r_np,
        'r_final_sym': r_sym,
        'r_diff': r_diff,
        'r_diff_norm_au': float(np.linalg.norm(r_diff)),
        'r_diff_norm_m': float(np.linalg.norm(r_diff) * AU_TO_M),
    }


