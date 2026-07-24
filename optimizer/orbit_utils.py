"""
Auxiliary functions for MEE (Mean Equinoctial Elements) and delta-v operations.
"""
import casadi as ca

# Fixed internal resolution for Kepler coast integration.
KEPLER_SUBSTEPS = 10

# ============================================================
# Pre-compiled Kepler coast function (created once, reused always)
# ============================================================

def _create_kepler_coast_function(n_kepler):
    """
    Create a pre-compiled Kepler coast function using CasADi's native RK integrator.

    To keep dt variable in Opti graphs, the ODE is time-scaled with a normalized
    integration horizon [0, 1], and dt is passed through parameters.

    Integrates: dL/dt = sqrt(μ*p) * (w/p)^2, with w = 1 + f*cos(L) + g*sin(L)
    
    Args:
        n_kepler: Number of RK4 sub-steps
    
    Returns:
        CasADi Function that propagates [L, p, f, g, mu, dt] -> L_new
    """
    L = ca.MX.sym('L')
    p = ca.MX.sym('p')
    f = ca.MX.sym('f')
    g = ca.MX.sym('g')
    mu = ca.MX.sym('mu')
    dt = ca.MX.sym('dt')

    # Robust guards to avoid NaN in invalid intermediate iterates.
    p_safe = ca.fmax(ca.fabs(p), 1e-10)
    w = 1.0 + f * ca.cos(L) + g * ca.sin(L)
    sqrt_arg = ca.fmax(mu * p_safe, 1e-12)
    dL_dt = ca.sqrt(sqrt_arg) * (w / p_safe) ** 2

    # Normalize integration window to [0,1] and scale ODE by dt.
    dae = {
        'x': L,
        'p': ca.vertcat(p, f, g, mu, dt),
        'ode': dt * dL_dt,
    }
    intg = ca.integrator(
        f'rk4_kepler_{n_kepler}',
        'rk',
        dae,
        {
            'number_of_finite_elements': int(n_kepler),
            'simplify': True,
        }
    )

    out = intg(x0=L, p=ca.vertcat(p, f, g, mu, dt))
    return ca.Function(f'kepler_coast_{n_kepler}', [L, p, f, g, mu, dt], [out['xf']])

# Cache for compiled functions (one per sub-step count)
_kepler_coast_cache = {}


def _dot_sym(a, b):
    """3D dot product helper for CasADi column vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross_sym(a, b):
    """3D cross product helper for CasADi column vectors."""
    return ca.vertcat(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def mee_to_rv_sym(x, mu):
    """Convert MEE state [p,f,g,h,k,L] to inertial Cartesian (r,v)."""
    p = x[0, 0]
    f = x[1, 0]
    g = x[2, 0]
    h = x[3, 0]
    k = x[4, 0]
    L = x[5, 0]

    cosL = ca.cos(L)
    sinL = ca.sin(L)
    w = 1 + f * cosL + g * sinL
    s2 = 1 + h * h + k * k
    p_safe = ca.fmax(ca.fabs(p), 1e-10)
    w_safe = ca.if_else(ca.fabs(w) < 1e-10, ca.sign(w) * 1e-10 + (1 - ca.fabs(ca.sign(w))) * 1e-10, w)

    fhat = ca.vertcat((1 - k * k + h * h) / s2, (2 * h * k) / s2, (-2 * k) / s2)
    ghat = ca.vertcat((2 * h * k) / s2, (1 + k * k - h * h) / s2, (2 * h) / s2)

    r = (p_safe / w_safe) * (cosL * fhat + sinL * ghat)
    v = ca.sqrt(mu / p_safe) * (-(g + sinL) * fhat + (f + cosL) * ghat)
    return r, v


def rv_to_mee_sym(r, v, mu):
    """Convert inertial Cartesian (r,v) to MEE state [p,f,g,h,k,L]."""
    # Radius and angular momentum
    r_norm = ca.norm_2(r)
    h_vec = _cross_sym(r, v)
    h_norm = ca.norm_2(h_vec)
    r_norm_safe = ca.fmax(r_norm, 1e-10)
    h_norm_safe = ca.fmax(h_norm, 1e-10)

    # Conic parameter and normal
    p = h_norm_safe * h_norm_safe / mu
    hhat = h_vec / h_norm_safe

    # Inclination-related variables
    den = 1 + hhat[2]
    den_safe = ca.if_else(ca.fabs(den) < 1e-10, ca.sign(den) * 1e-10 + (1 - ca.fabs(ca.sign(den))) * 1e-10, den)
    h = -hhat[1] / den_safe
    k = hhat[0] / den_safe
    s2 = 1 + h * h + k * k

    # Eccentricity auxiliar variables
    fhat = ca.vertcat((1 - k * k + h * h) / s2, (2 * h * k) / s2, (-2 * k) / s2)
    ghat = ca.vertcat((2 * h * k) / s2, (1 + k * k - h * h) / s2, (2 * h) / s2)

    # True longitude
    x_plane = _dot_sym(r, fhat)
    y_plane = _dot_sym(r, ghat)
    L = ca.atan2(y_plane, x_plane)

    # Eccentricity-related variables
    e_vec = _cross_sym(v, h_vec) / mu - r / r_norm_safe
    f = _dot_sym(e_vec, fhat)
    g = _dot_sym(e_vec, ghat)

    return ca.vertcat(p, f, g, h, k, L)


def get_kepler_substeps():
    """Return the fixed number of RK finite elements used in Kepler coasts."""
    return KEPLER_SUBSTEPS


def kepler_coast_sym(x, dt, mu):
    """
    Propagate MEE state under Keplerian motion using CasADi native RK.
    
    Key insight: Under pure Keplerian motion, the first 5 MEE elements
    (p, f, g, h, k) are constant. Only L (true longitude) changes.
    
    Uses native CasADi RK integration with numerical protection:
    dL/dt = sqrt(μ*p) * (w/p)²
    where w = 1 + f*cos(L) + g*sin(L)
    
    The function is compiled ONCE and cached to avoid recompilation overhead,
    while maintaining full symbolic differentiability for the optimizer.
    
    Args:
        x: 6x1 CasADi vector [p, f, g, h, k, L]
        dt: Time step
        mu: Gravitational parameter
    Returns:
        6x1 CasADi vector with propagated state [p, f, g, h, k, L_new]
    """
    # Extract state
    p  = x[0,0]
    f  = x[1,0]
    g  = x[2,0]
    h  = x[3,0]
    k  = x[4,0]
    L  = x[5,0]
    
    # Get or create compiled function (cached to avoid recompilation)
    n_kepler = KEPLER_SUBSTEPS
    if n_kepler not in _kepler_coast_cache:
        _kepler_coast_cache[n_kepler] = _create_kepler_coast_function(n_kepler)
    
    kepler_func = _kepler_coast_cache[n_kepler]
    
    # Evaluate compiled function
    L_new = kepler_func(L, p, f, g, mu, dt)
    
    return ca.vertcat(p, f, g, h, k, L_new)



def apply_dv_sym(x, dv, mu):
    """
    Apply a delta-v impulse to MEE state using Gauss equations.
    
    Args:
        x: 6x1 CasADi vector [p, f, g, h, k, L]
        dv: 3x1 CasADi vector [dvR, dvT, dvN] in RTN frame
        mu: Gravitational parameter
    
    Returns:
        6x1 CasADi vector with updated state
    """
    r, v = mee_to_rv_sym(x, mu)

    # Radial
    r_norm = ca.fmax(ca.norm_2(r), 1e-10)
    rhat = r / r_norm

    # Out-of-plane
    hhat = _cross_sym(r, v)
    h_norm = ca.fmax(ca.norm_2(hhat), 1e-10)
    hhat = hhat / h_norm

    # Tangential
    that = _cross_sym(hhat, rhat)

    # Delta-v in inertial frame
    dv_inertial = dv[0] * rhat + dv[1] * that + dv[2] * hhat

    # Velocity increment
    v_post = v + dv_inertial

    return rv_to_mee_sym(r, v_post, mu)


def propagate_L(MEE0, T, mu):
    """
    Propagate target state's true longitude over transfer time.
    
    Args:
        p, f, g, h, k: Target MEE elements
        L0: Initial true longitude
        T: Transfer time
        mu: Gravitational parameter
    Returns:
        Final true longitude as CasADi variable
    """
    x_tgt = ca.vertcat(MEE0)
    xf_tgt = kepler_coast_sym(x_tgt, T, mu)

    return xf_tgt[5,0]

