"""P1-C-small: high-accuracy tensor-grid DP reference (H = 3), review sec.7-9.

Bellman recursion on the buffer grid [-L, L]^{H+1} with Gauss-Hermite
expectation, action-grid minimisation over the box, and 1-D interpolation in
the new first coordinate (the remaining coordinates shift exactly on-grid).
This is a REFINED NUMERICAL DP REFERENCE, not an exact oracle: errors from
state-grid interpolation, GH quadrature, the action grid, and the finite
domain. Refinement audit (n_x, n_GH, n_u) is therefore part of the artifact.
"""
import time
import numpy as np
from scipy.ndimage import map_coordinates
from .config import load_config

def gh_nodes(n):
    x, w = np.polynomial.hermite.hermgauss(n)
    return np.sqrt(2.0)*x, w/np.sqrt(np.pi)

def _interp0(Vn, Xp, xg):
    n = len(xg); dx = xg[1] - xg[0]
    pos = np.clip((Xp - xg[0])/dx, 0.0, n - 1 - 1e-9)
    lo = pos.astype(np.int64); fr = pos - lo
    I0 = lo[:, None, None, :]
    A1 = np.arange(n)[:, None, None, None]
    A2 = np.arange(n)[None, :, None, None]
    A3 = np.arange(n)[None, None, :, None]
    Va = Vn[I0, A1, A2, A3]; Vb = Vn[np.minimum(I0 + 1, n - 1), A1, A2, A3]
    f = fr[:, None, None, :]
    return (1 - f)*Va + f*Vb

def dp_reference(cfg, n_x=25, n_gh=5, n_u=21, L=3.0, bounds=None):
    """Backward tensor-grid DP for the H=3 buffer. Returns value tables,
    greedy action tables (uint8 indices), and grids."""
    p = cfg["params"]; h, N, H = cfg["h"], cfg["N"], cfg["H"]
    assert H == 3, "tensor-grid DP is for the small variant only"
    a, ad, b = p["a"], p["ad"], p["b"]
    s0, cx, cy, gu = p["s0"], p["cx"], p["cy"], p["gu"]
    Q, R, QT = p["Q"], p["R"], p["QT"]
    lo_u, hi_u = bounds if bounds is not None else cfg["bounds"]
    xg = np.linspace(-L, L, n_x); ug = np.linspace(lo_u, hi_u, n_u)
    gx, gw = gh_nodes(n_gh)
    V = np.broadcast_to(0.5*QT*(xg - cfg["xtar"])**2, (n_x,)*4)[..., None]*0
    V = np.tile((0.5*QT*(xg - cfg["xtar"])**2)[:, None, None, None], (1, n_x, n_x, n_x))
    Vs = [None]*(N+1); Vs[N] = V.copy()
    pol = [None]*N
    t0 = time.perf_counter()
    for k in range(N-1, -1, -1):
        run0 = 0.5*Q*h*(xg - cfg["xref"][k])**2
        best = None; bidx = None
        for iu, u in enumerate(ug):
            drift = xg[:, None]*(1 + a*h) + h*(ad*xg[None, :] + b*u)
            sig = s0 + cx*xg[:, None] + cy*xg[None, :] + gu*u
            EV = np.zeros((n_x,)*4)
            for g, wgt in zip(gx, gw):
                EV += wgt*_interp0(V, drift + np.sqrt(h)*sig*g, xg)
            tot = run0[:, None, None, None] + 0.5*R*h*u*u + EV
            if best is None:
                best, bidx = tot, np.zeros(tot.shape, dtype=np.uint8)
            else:
                m = tot < best
                best = np.where(m, tot, best); bidx = np.where(m, iu, bidx)
        V = best; Vs[k] = V.copy(); pol[k] = bidx
        assert np.isfinite(V).all()
    t_dp = time.perf_counter() - t0
    return dict(V=Vs, pol=pol, xg=xg, ug=ug, L=L, n_x=n_x, n_gh=n_gh, n_u=n_u,
                bounds=(lo_u, hi_u), runtime=t_dp)

def dp_value_at(dp, k, z):
    c = (np.asarray(z) - dp["xg"][0])/(dp["xg"][1] - dp["xg"][0])
    return float(map_coordinates(dp["V"][k], c[:, None], order=1, mode="nearest")[0])

def dp_action_at(dp, k, z):
    c = (np.asarray(z) - dp["xg"][0])/(dp["xg"][1] - dp["xg"][0])
    ua = dp["ug"][dp["pol"][k]]
    return float(map_coordinates(ua, c[:, None], order=1, mode="nearest")[0])

def dp_policy(dp):
    def pol(k, Z):
        return np.array([dp_action_at(dp, k, z) for z in np.atleast_2d(Z)])
    return pol
