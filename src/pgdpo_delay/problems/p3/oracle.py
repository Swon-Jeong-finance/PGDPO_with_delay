"""P3-R fine-grid 2D HJB numerical reference on the (I, M) lift.

Only an instance published by the full certification workflow is a frozen
reference; this implementation is never described as an "exact oracle".

HJB (cost half-convention, minimisation):
  -V_t = min_{u in [0,1]} { (c_I/2)I^2 + (R/2)u^2
          + [beta(1-I/N)M - gamma I - b u I] V_I + rho (I - M) V_M
          + (1/2) sigma0^2 (1-eta u)^2 I^2 V_II }.

DISCRETE control selection (review 2026-08-07 sec.3.2, H1.1): the monotone
upwind operator's control block is a PIECEWISE quadratic in u -- the drift
b_I(u) = base - bIu changes sign at u_sw = base/(bI), switching between the
forward difference D+V (b_I > 0) and the backward difference D-V (b_I < 0).
`discrete_min_u` minimises that branch-aware objective EXACTLY by comparing
the endpoints {0, 1}, the switch point u_sw, and the per-branch clipped
quadratic vertices, all evaluated with the full upwind discrete Hamiltonian
(continuous at u_sw because the drift term vanishes there). The continuous
central-derivative formula `pointwise_min_u` is retained ONLY as the
continuous-algebra fixture (brute-force gate) -- it is NOT the solver's
minimiser.

Scheme: explicit positive-coefficient upwind, central second difference for
the I-diffusion, CFL-limited substeps. Boundary convention (sec.3.4): the
advective characteristics point inward on every boundary for all u in [0,1]
(validated at solve time), and the NONDEGENERATE upper-I diffusion boundary
is closed by a linear-extrapolation ghost (V_II = 0 there) -- an explicit
artificial truncation convention certified by the matched-spacing domain-
expansion gate, NOT "no boundary condition". M has no diffusion.

Per-solve diagnostics accumulated over every solver surface, from the
terminal surface through the final t=0 endpoint (sec.3.7): global curvature
min A2, nonpositive-curvature fraction, V_II range.
"""
import time
import numpy as np
from scipy.ndimage import map_coordinates

P3_API_VERSION = "p3-oracle-v3-reference-contract"


def pointwise_min_u(params, I, V_I, V_II, bounds=(0.0, 1.0)):
    """CONTINUOUS-algebra minimiser (central derivatives): quadratic vertex
    with endpoint comparison under the curvature guard. Kept as the algebra
    fixture; the solver uses `discrete_min_u`."""
    b, eta, s0, R = params["b"], params["eta_sigma"], params["sigma0"], params["R"]
    lo, hi = bounds
    s2I2 = (s0*I)**2
    A2 = R + (eta**2)*s2I2*V_II
    num = b*I*V_I + eta*s2I2*V_II
    ok = A2 > 1e-14
    u_vtx = np.clip(np.where(ok, num/np.where(ok, A2, 1.0), 0.0), lo, hi)
    f = lambda u: 0.5*A2*u*u - num*u
    u_end = np.where(f(np.full_like(u_vtx, hi)) < 0.0, hi, lo)
    return np.where(ok, u_vtx, u_end)


def upwind_control_objective(params, I, base, Dp, Dm, DII, u):
    """Full branch-aware upwind control block at u (state running cost
    dropped -- it is u-independent). Vectorised over broadcastable grids."""
    b, eta, s0, R = params["b"], params["eta_sigma"], params["sigma0"], params["R"]
    bIu = base - b*I*u
    s2I2 = (s0*I)**2
    return (0.5*R*u*u + np.maximum(bIu, 0.0)*Dp + np.minimum(bIu, 0.0)*Dm
            + 0.5*s2I2*(1.0 - eta*u)**2*DII)


def discrete_min_u(params, I, base, Dp, Dm, DII, lo=0.0, hi=1.0):
    """EXACT minimiser of the branch-aware upwind control block (H1.1;
    empty-branch masking per re-review 2026-08-07 sec.4). Candidates:
    endpoints, drift switch point, per-branch vertices restricted to their
    FEASIBLE branch intervals -- when a branch interval is empty (switch
    point outside the box) its vertex candidate collapses to an endpoint
    that is already enumerated, never to an out-of-box value (np.clip with
    lower > upper would manufacture infeasible candidates). Duplicates are
    harmless because every candidate is scored with the exact piecewise
    objective."""
    b, eta, s0, R = params["b"], params["eta_sigma"], params["sigma0"], params["R"]
    bI = b*I
    s2I2 = (s0*I)**2
    A2 = R + (eta**2)*s2I2*DII
    lin_dif = eta*s2I2*DII
    ok = A2 > 1e-14
    safe = np.where(ok, A2, 1.0)
    v_pos = np.where(ok, (bI*Dp + lin_dif)/safe, lo)     # b_I > 0 branch
    v_neg = np.where(ok, (bI*Dm + lin_dif)/safe, hi)     # b_I < 0 branch
    with np.errstate(divide="ignore", invalid="ignore"):
        usw = np.where(bI > 1e-14,
                       base/np.where(bI > 1e-14, bI, 1.0),
                       np.where(base >= 0.0, np.inf, -np.inf))
    pos_hi = np.minimum(usw, hi)                          # branch+ = [lo, pos_hi]
    neg_lo = np.maximum(usw, lo)                          # branch- = [neg_lo, hi]
    c_pos = np.where(pos_hi >= lo,
                     np.minimum(np.maximum(v_pos, lo), pos_hi), lo)
    c_neg = np.where(neg_lo <= hi,
                     np.minimum(np.maximum(v_neg, neg_lo), hi), hi)
    cands = np.stack([
        np.full_like(base, lo),
        np.full_like(base, hi),
        np.clip(usw, lo, hi),
        c_pos,
        c_neg,
    ])
    fs = upwind_control_objective(params, I, base, Dp, Dm, DII, cands)
    j = fs.argmin(axis=0)
    return np.take_along_axis(cands, j[None], 0)[0]


def _derivatives(V, dI):
    """Central V_I (one-sided at the I-boundaries; algebra-fixture use) and
    central V_II with linear-extrapolation ghosts (V_II = 0 at both ends; at
    I = 0 the diffusion coefficient vanishes anyway, at I = I_max this is the
    explicit truncation closure)."""
    V_I = np.empty_like(V); V_II = np.zeros_like(V)
    V_I[1:-1] = (V[2:] - V[:-2])/(2*dI)
    V_I[0] = (V[1] - V[0])/dI
    V_I[-1] = (V[-1] - V[-2])/dI
    V_II[1:-1] = (V[2:] - 2*V[1:-1] + V[:-2])/dI**2
    return V_I, V_II


def _updiffs(V, dI):
    """Forward/backward first differences with inward-only boundary rows
    (fwd at I_max and bwd at I = 0 are never used: drift signs there are
    validated inward) and the ghosted second difference."""
    fwd = np.zeros_like(V); bwd = np.zeros_like(V); DII = np.zeros_like(V)
    fwd[:-1] = (V[1:] - V[:-1])/dI
    bwd[1:] = (V[1:] - V[:-1])/dI
    DII[1:-1] = (V[2:] - 2*V[1:-1] + V[:-2])/dI**2
    return fwd, bwd, DII


def solve_hjb(cfg, n_I=None, n_M=None, cfl_safety=None, I_max=None,
              store_policy=True, store_value=False):
    p = cfg["params"]; dt_sim, N = cfg["dt"], cfg["N"]
    hj = cfg["hjb"]
    n_I = n_I or hj["n_I"]; n_M = n_M or hj["n_M"]
    cfl = cfl_safety or hj["cfl_safety"]
    I_max = I_max or hj["I_max"]; M_max = I_max
    Ig = np.linspace(0.0, I_max, n_I); Mg = np.linspace(0.0, M_max, n_M)
    dI, dM = Ig[1]-Ig[0], Mg[1]-Mg[0]
    II, MM = Ig[:, None], Mg[None, :]

    bI_at_max = p["beta"]*(1.0 - I_max/p["Npop"])*Mg - p["gamma"]*I_max
    if not (bI_at_max < 0).all():
        raise ValueError("I_max boundary drift not inward: enlarge I_max")

    Dmax = 0.5*(p["sigma0"]*I_max)**2
    base = p["beta"]*(1.0 - II/p["Npop"])*MM - p["gamma"]*II   # u-independent
    bI_max = np.abs(base).max() + p["b"]*I_max
    bM_max = p["rho"]*M_max
    dt_hjb = cfl/(2*Dmax/dI**2 + bI_max/dI + bM_max/dM)
    n_sub = max(1, int(np.ceil(dt_sim/dt_hjb)))
    dt_h = dt_sim/n_sub

    V = 0.5*p["c_T"]*II**2 + 0.0*MM
    V = np.broadcast_to(V, (n_I, n_M)).copy()
    pol = [None]*N if store_policy else None
    Vs = [None]*(N+1) if store_value else None   # time-indexed snapshots
    if store_value:
        Vs[N] = V.astype(np.float32)             # terminal surface
    run_I = 0.5*p["c_I"]*II**2
    bM = p["rho"]*(II - MM)
    s2I2 = (p["sigma0"]*II)**2
    eta = p["eta_sigma"]
    A2_min = np.inf; A2_nonpos = 0; n_nodes = 0
    VII_min, VII_max = np.inf, -np.inf
    t0 = time.perf_counter()
    for k in range(N-1, -1, -1):
        for s in range(n_sub):
            fwd, bwd, DII = _updiffs(V, dI)
            u = discrete_min_u(p, II, base, fwd, bwd, DII)
            A2 = p["R"] + (eta**2)*s2I2*DII
            A2_min = min(A2_min, float(A2.min()))
            A2_nonpos += int((A2 <= 0).sum()); n_nodes += A2.size
            VII_min = min(VII_min, float(DII.min()))
            VII_max = max(VII_max, float(DII.max()))
            bIu = base - p["b"]*II*u
            fwdM = np.zeros_like(V); bwdM = np.zeros_like(V)
            fwdM[:, :-1] = (V[:, 1:] - V[:, :-1])/dM
            bwdM[:, 1:] = (V[:, 1:] - V[:, :-1])/dM
            up_M = np.maximum(bM, 0.0)*fwdM + np.minimum(bM, 0.0)*bwdM
            V = V + dt_h*(run_I + 0.5*p["R"]*u*u
                          + np.maximum(bIu, 0.0)*fwd + np.minimum(bIu, 0.0)*bwd
                          + up_M
                          + 0.5*s2I2*(1.0 - eta*u)**2*DII)
            if not np.isfinite(V).all():
                raise FloatingPointError(f"HJB value not finite at k={k}, sub={s}")
        if store_policy:
            fwd, bwd, DII = _updiffs(V, dI)
            pol[k] = discrete_min_u(p, II, base, fwd, bwd, DII).astype(np.float32)
        if store_value:
            Vs[k] = V.astype(np.float32)
    runtime = time.perf_counter() - t0

    # Every pre-update solver surface is audited inside the substep loop.  The
    # final update produces V(0), for which there is no following substep, so
    # audit that endpoint explicitly as well.  This makes the ``global`` A2
    # label and V_II range cover all N*n_sub + 1 solver surfaces without
    # changing the HJB value or policy recursion.
    _, _, DII = _updiffs(V, dI)
    A2 = p["R"] + (eta**2)*s2I2*DII
    A2_min = min(A2_min, float(A2.min()))
    A2_nonpos += int((A2 <= 0).sum()); n_nodes += A2.size
    VII_min = min(VII_min, float(DII.min()))
    VII_max = max(VII_max, float(DII.max()))

    return dict(V0=V, pol=pol, Vs=Vs, Ig=Ig, Mg=Mg, n_I=n_I, n_M=n_M, I_max=I_max,
                dt_hjb=dt_h, n_sub=n_sub, runtime=runtime,
                A2_min_global=A2_min,
                A2_nonpos_frac=A2_nonpos/max(1, n_nodes),
                V_II_range=(VII_min, VII_max),
                curvature_diagnostic_scope="all_solver_surfaces_including_t0",
                curvature_diagnostic_surface_count=N*n_sub + 1)


def value_at(hjb, I, M):
    """Bilinear V(0, I, M) readout (clamped to the grid)."""
    Ig, Mg = hjb["Ig"], hjb["Mg"]
    ci = np.clip((np.atleast_1d(I) - Ig[0])/(Ig[1]-Ig[0]), 0, len(Ig)-1)
    cm = np.clip((np.atleast_1d(M) - Mg[0])/(Mg[1]-Mg[0]), 0, len(Mg)-1)
    return map_coordinates(hjb["V0"], np.stack([ci, cm]), order=1, mode="nearest")


def hjb_policy(hjb):
    """Bilinear feedback policy readout pol(k, I, M) (clamped to grid).
    Smooth-trajectory readout; ACTIVE-SET statistics must use regime labels
    with the dtype-aware tolerance (evaluate-layer contract)."""
    Ig, Mg = hjb["Ig"], hjb["Mg"]
    def pol(k, I, M):
        ci = np.clip((np.atleast_1d(I) - Ig[0])/(Ig[1]-Ig[0]), 0, len(Ig)-1)
        cm = np.clip((np.atleast_1d(M) - Mg[0])/(Mg[1]-Mg[0]), 0, len(Mg)-1)
        return map_coordinates(hjb["pol"][k], np.stack([ci, cm]),
                               order=1, mode="nearest")
    return pol
