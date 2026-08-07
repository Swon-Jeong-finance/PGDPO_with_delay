"""P1-C small variant: PRELIMINARY tensor-grid numerical DP implementation
under refinement (review 2026-08-07: NOT yet a certified high-accuracy
reference; do not call it an exact constrained oracle).

Improvements in this revision:
  * FINAL Bellman step analytic: E[V_N] under the terminal quadratic is
    computed in closed form (no interpolation), so with the parabolic action
    refinement the k = N-1 policy matches the exact clipped last-step action
    to machine precision (review sec.8.1 gate).
  * Sub-grid action refinement: parabolic vertex through the three
    bracketing action-grid values (exact for quadratic-in-u objectives),
    always clipped to the box; boundary arg-minima keep the bound exactly.
  * Policy tables store ACTIONS (float32), not uint8 indices.
  * Role split for policy readout (review sec.10):
      dp_action_label_at        nearest-node action  -> active-set labels
      dp_action_interpolated_at multilinear action   -> smooth trajectories
  * Grid-quadrature clipping diagnostic: mean continuation OOB fraction
    over the full Bellman tensor/action/GH grid; NOT a rollout or
    domain-certification metric (minfix review sec.6).
Remaining refinement track: state-interpolation upgrade (convexity-preserving)
and the (n_x, n_gh, n_u, L) refinement ladder with accuracy gates.
Ladder status (2026-08-07): implementation DEFERRED (appendix-audit only, not
on the paper critical path); the binding spec is
docs/decisions/PGDPO_P1C_code_and_DP_ladder_review_20260807.md (sec.11 phase order, matched-dx
domain rungs r2D/r3D, explicit int8 regime table, reoptimized readout).
"""
import time
import numpy as np
from scipy.ndimage import map_coordinates
from .config import load_config

def gh_nodes(n):
    x, w = np.polynomial.hermite.hermgauss(n)
    return np.sqrt(2.0)*x, w/np.sqrt(np.pi)

def exact_last_step_action(cfg, x0, xH):
    """Exact box-constrained minimiser of the final Bellman step (terminal
    quadratic; review sec.9.1). Strong unit test for the DP engine."""
    p = cfg["params"]; h = cfg["h"]
    mean0 = (1 + p["a"]*h)*x0 + h*p["ad"]*xH
    sigma0 = p["s0"] + p["cx"]*x0 + p["cy"]*xH
    denom = h*p["R"] + p["QT"]*((h*p["b"])**2 + h*p["gu"]**2)
    numer = p["QT"]*(h*p["b"]*(mean0 - cfg["xtar"]) + h*p["gu"]*sigma0)
    return np.clip(-numer/denom, *cfg["bounds"])

def _interp0(Vn, Xp, xg, oob_acc=None):
    n = len(xg); dx = xg[1] - xg[0]
    pos = (Xp - xg[0])/dx
    if oob_acc is not None:
        oob_acc.append(float(((pos < 0) | (pos > n - 1)).mean()))
    pos = np.clip(pos, 0.0, n - 1 - 1e-9)
    lo = pos.astype(np.int64); fr = pos - lo
    I0 = lo[:, None, None, :]
    A1 = np.arange(n)[:, None, None, None]
    A2 = np.arange(n)[None, :, None, None]
    A3 = np.arange(n)[None, None, :, None]
    Va = Vn[I0, A1, A2, A3]; Vb = Vn[np.minimum(I0 + 1, n - 1), A1, A2, A3]
    f = fr[:, None, None, :]
    return (1 - f)*Va + f*Vb

def dp_reference(cfg, n_x=None, n_gh=None, n_u=None, L=None, bounds=None):
    p = cfg["params"]; h, N, H = cfg["h"], cfg["N"], cfg["H"]
    if H != 3:
        raise ValueError("tensor-grid DP is for the small (H=3) variant only")
    dp_cfg = cfg.get("dp", {})
    n_x = n_x or dp_cfg.get("n_x"); n_gh = n_gh or dp_cfg.get("n_gh")
    n_u = n_u or dp_cfg.get("n_u"); L = L or dp_cfg.get("L")
    a, ad, b = p["a"], p["ad"], p["b"]
    s0, cx, cy, gu = p["s0"], p["cx"], p["cy"], p["gu"]
    Q, R, QT = p["Q"], p["R"], p["QT"]
    lo_u, hi_u = bounds if bounds is not None else cfg["bounds"]
    xg = np.linspace(-L, L, n_x); ug = np.linspace(lo_u, hi_u, n_u)
    du = ug[1] - ug[0]
    gx, gw = gh_nodes(n_gh)
    V = np.tile((0.5*QT*(xg - cfg["xtar"])**2)[:, None, None, None], (1, n_x, n_x, n_x))
    Vs = [None]*(N+1); Vs[N] = V.copy()
    pol = [None]*N
    oob = []
    t0 = time.perf_counter()
    for k in range(N-1, -1, -1):
        run0 = 0.5*Q*h*(xg - cfg["xref"][k])**2
        tots = np.empty((n_u,) + (n_x,)*4, dtype=np.float32)
        for iu, u in enumerate(ug):
            drift = xg[:, None]*(1 + a*h) + h*(ad*xg[None, :] + b*u)
            sig = s0 + cx*xg[:, None] + cy*xg[None, :] + gu*u
            if k == N - 1:
                # analytic terminal expectation: no interpolation at the last step
                ev2 = 0.5*QT*((drift - cfg["xtar"])**2 + h*sig**2)
                EV = np.broadcast_to(ev2[:, None, None, :], (n_x,)*4).copy()
            else:
                EV = np.zeros((n_x,)*4)
                for g, wgt in zip(gx, gw):
                    EV += wgt*_interp0(V, drift + np.sqrt(h)*sig*g, xg, oob)
            tots[iu] = run0[:, None, None, None] + 0.5*R*h*u*u + EV
        j = tots.argmin(axis=0)
        # parabolic sub-grid refinement through the 3-point stencil AT the
        # argmin; boundary arg-minima use the one-sided stencil so interior
        # minimisers within one grid cell of a bound are NOT snapped to it
        # (active-set labels depend on this; exact for quadratic-in-u steps).
        jc = np.clip(j, 1, n_u - 2)                      # stencil centre
        f0 = np.take_along_axis(tots, jc[None], 0)[0].astype(np.float64)
        fm = np.take_along_axis(tots, (jc - 1)[None], 0)[0].astype(np.float64)
        fp = np.take_along_axis(tots, (jc + 1)[None], 0)[0].astype(np.float64)
        curv = fp - 2*f0 + fm
        ok = curv > 1e-14
        shift = np.where(ok, 0.5*du*(fm - fp)/np.where(ok, curv, 1.0), 0.0)
        vertex = ug[jc] + np.clip(shift, -1.5*du, 1.5*du)
        u_ref = np.where(ok, np.clip(vertex, lo_u, hi_u), ug[j])
        v_par = f0 - 0.125*(fm - fp)**2/np.where(ok, curv, 1.0)
        fj = np.take_along_axis(tots, j[None], 0)[0].astype(np.float64)
        # value: parabola minimum where valid AND the vertex stays in the box;
        # otherwise the grid minimum (which sits at the active bound)
        in_box = (vertex >= lo_u) & (vertex <= hi_u)
        V = np.where(ok & in_box, np.minimum(v_par, fj), fj)
        Vs[k] = V.copy(); pol[k] = u_ref.astype(np.float32)
        if not np.isfinite(V).all():
            raise FloatingPointError(f"DP value not finite at k={k}")
    t_dp = time.perf_counter() - t0
    return dict(V=Vs, pol=pol, xg=xg, ug=ug, L=L, n_x=n_x, n_gh=n_gh, n_u=n_u,
                bounds=(lo_u, hi_u), runtime=t_dp,
                # DIAGNOSTIC ONLY (review 2026-08-07 sec.4.7): uniform-weight
                # continuation clipping over the full Bellman tensor grid at
                # ALL candidate actions/GH nodes. NOT an audit-bank or rollout
                # boundary metric; never use it as a domain-certification (G4)
                # input or compare it directly across different L.
                grid_quadrature_oob_frac=float(np.mean(oob)) if oob else 0.0)

def dp_value_at(dp, k, z):
    c = (np.asarray(z) - dp["xg"][0])/(dp["xg"][1] - dp["xg"][0])
    return float(map_coordinates(dp["V"][k], c[:, None], order=1, mode="nearest")[0])

def dp_action_label_at(dp, k, z):
    """Nearest-node action: use for active-set labels, occupancy, switching."""
    idx = np.clip(np.rint((np.asarray(z) - dp["xg"][0])/(dp["xg"][1] - dp["xg"][0])),
                  0, dp["n_x"] - 1).astype(int)
    return float(dp["pol"][k][tuple(idx)])

def dp_action_interpolated_at(dp, k, z):
    """Multilinear action: smooth trajectory proxy ONLY (can average across
    regimes; never use for active-set statistics)."""
    c = (np.asarray(z) - dp["xg"][0])/(dp["xg"][1] - dp["xg"][0])
    return float(map_coordinates(dp["pol"][k], c[:, None], order=1, mode="nearest")[0])

def dp_action_at(dp, k, z):
    """Deprecated alias for dp_action_interpolated_at (kept for audits)."""
    return dp_action_interpolated_at(dp, k, z)

def dp_policy(dp, label=True):
    fn = dp_action_label_at if label else dp_action_interpolated_at
    def pol(k, Z):
        return np.array([fn(dp, k, z) for z in np.atleast_2d(Z)])
    return pol
