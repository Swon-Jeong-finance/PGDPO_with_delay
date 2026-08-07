"""P1 controlled-diffusion point-delay LQ: exact same-grid discrete oracles.

Discrete model (Euler, same grid as simulator), h = dt, H = delay taps:
  X_{k+1} = X_k + h(a X_k + a_d X_{k-H} + b u_k)
            + (s0 + c_x X_k + c_y X_{k-H} + g_u u_k) dW_k,
  cost  sum_k h[ Q/2 (X_k - r_k)^2 + R/2 u_k^2 ] + Q_T/2 (X_N - xtar)^2.

Buffer state Z_k = (X_k, ..., X_{k-H}) in R^{H+1}:
  Z_{k+1} = A Z_k + B u_k + (C Z_k + D u_k + Sg) dW_k,
  A = E + e1 r^T (down-shift + first row), C = e1 c^T (rank one), B,D,Sg prop to e1.

TWO DISTINCT second-order objects (never conflate):
  Pval : value-function Hessian D^2_zz V_k. Generalized Riccati WITH the
         Schur re-optimisation term -K^T Lam^{-1} K. (Lam/h = R + g_u^2 Pval_11 + O(h)
         is the Bellman control curvature, NOT the SMP recovery denominator.)
  G_ol : detached fixed-control OL-BPTT expected Hessian of the continuation
         cost. NO Schur term:  G_k = hQ e1e1^T + A^T G' A + h C^T G' C.
         Pi_k := (G_ol[k])_{11} is the SMP effective-curvature target.
For P1 the state Jacobian of the transition, A + C dW, is control-free, so G_ol
is deterministic and policy-independent; zeta_k = q_{k+1|k} - Pi_k * sigma_k^ref
is generally nonzero. The two recovery forms return the SAME action because
zeta compensates:  (R + g_u^2 Pi) u* = -(b p + g_u zeta + g_u Pi sigma_bar)
is an algebraic re-coordinatisation of  R u* + b p_{k+1|k} + g_u q_{k+1|k} = 0.

Exactness needs only E_k[dW]=0, E_k[dW^2]=h (no Gaussian density). Two-point
noise dW = +-sqrt(h) is therefore an exact quadrature for ALGEBRA CHECKS ONLY;
paper rollouts and estimator-variance studies must keep Gaussian increments.

Cost: structured (shift + rank-one) products give O(H^2) per step, O(N H^2)
total; a dense fallback (O(N H^3)) is kept for verification.

P1-C note: the final one-step quadratic subproblem is obtained by clipping,
but backward propagation through the constrained value destroys the global
quadratic form; the full P1-C oracle is the low-dimensional tensor-grid DP.

Stage-II convention (Path A): the deployed PGDPO decoder consumes the
manuscript-level current gradient p_k; the one-step conditional p_{k+1|k}
is an exact same-grid Euler-FOC diagnostic only. Their finite-h action
difference is an h-refinement diagnostic, not an estimator error.
"""
ORACLE_API_VERSION = "p1-v3-pcur-pnext"

import numpy as np

# ---------------------------------------------------------------- structure
from pgdpo_delay.core.structured import AtPA, Ats, AtPe1
def coeffs(params, H, h):
    a, ad, b, s0, cx, cy, gu = (params[k] for k in ("a","ad","b","s0","cx","cy","gu"))
    n = H + 1
    r = np.zeros(n); r[0] = 1 + a*h; r[H] += ad*h          # first row of A
    c = np.zeros(n); c[0] = cx; c[H] += cy                  # first row of C
    return n, r, c

def build_dense(params, H, h):
    a, ad, b, s0, cx, cy, gu = (params[k] for k in ("a","ad","b","s0","cx","cy","gu"))
    n, r, c = coeffs(params, H, h)
    A = np.zeros((n, n)); A[0] = r
    for i in range(1, n): A[i, i-1] = 1.0
    C = np.zeros((n, n)); C[0] = c
    B = np.zeros(n); B[0] = b*h
    D = np.zeros(n); D[0] = gu
    Sg = np.zeros(n); Sg[0] = s0
    return A, B, C, D, Sg

# ---------------------------------------------------------------- oracles
def riccati(params, H, h, N, xref, xtar):
    """Optimal-control oracle: value quadratic V_k = 1/2 z'Pval z + s'z + c
    and affine feedback u* = F z + f. Exact for the same-grid Euler problem."""
    Q, R, QT = params["Q"], params["R"], params["QT"]
    b, s0, gu = params["b"], params["s0"], params["gu"]
    assert R > 0 and Q >= 0 and QT >= 0
    n, r, c = coeffs(params, H, h)
    e1 = np.zeros(n); e1[0] = 1.0
    P = QT*np.outer(e1, e1); s = -QT*xtar*e1; cc = 0.5*QT*xtar**2
    Pv, sv, cv = [None]*(N+1), [None]*(N+1), [None]*(N+1)
    F, f, Lams = [None]*N, [None]*N, [None]*N
    Pv[N], sv[N], cv[N] = P, s, cc
    for k in range(N-1, -1, -1):
        P11 = P[0, 0]
        Lam = h*R + (b*h)**2*P11 + h*gu*gu*P11
        K   = b*h*AtPe1(P, r) + h*gu*P11*c
        kap = b*h*s[0] + h*gu*s0*P11
        Pn = h*Q*np.outer(e1, e1) + AtPA(P, r) + h*P11*np.outer(c, c) - np.outer(K, K)/Lam
        Pn = 0.5*(Pn + Pn.T)
        sn = -h*Q*xref[k]*e1 + Ats(s, r) + h*s0*P11*c - K*(kap/Lam)
        cn = cc + 0.5*h*s0*s0*P11 + 0.5*h*Q*xref[k]**2 - 0.5*kap*kap/Lam
        F[k], f[k], Lams[k] = -K/Lam, -kap/Lam, Lam
        P, s, cc = Pn, sn, cn
        Pv[k], sv[k], cv[k] = P, s, cc
    return dict(Pval=Pv, s=sv, c=cv, F=F, f=f, Lam=Lams)

def detached_curvature(params, H, h, N):
    """Expected detached fixed-control OL-BPTT Hessian G_ol (Lyapunov-type, no
    Schur term). Policy-independent in P1 because A + C dW is control-free.
    Pi_k = G_ol[k][0,0]."""
    Q, QT = params["Q"], params["QT"]
    n, r, c = coeffs(params, H, h)
    e1 = np.zeros(n); e1[0] = 1.0
    G = [None]*(N+1)
    G[N] = QT*np.outer(e1, e1)
    for k in range(N-1, -1, -1):
        Gk = h*Q*np.outer(e1, e1) + AtPA(G[k+1], r) + h*G[k+1][0, 0]*np.outer(c, c)
        G[k] = 0.5*(Gk + Gk.T)
    return G

def exact_recovery_inputs(k, z, params, H, h, orc, G, xref):
    """Exact recovery-input oracle under the optimal policy. Returns a dict:
      u      : optimal action u*_k
      p_cur  : manuscript-level current gradient  p_k   = e1'(Pval_k z + s_k)
      p_nxt  : one-step conditional input p_{k+1|k} = e1'(Pval_{k+1} m + s_{k+1})
               (exact Euler-FOC diagnostic only; NOT the Stage-II input)
      q      : q_{k+1|k},  Pi : detached curvature,  zeta : q - Pi sigma*
      u_rec  : anchored reconstruction from (p_nxt, zeta, Pi)  (== u exactly)
    """
    b, s0, gu, R = params["b"], params["s0"], params["gu"], params["R"]
    A, B, C, D, Sg = build_dense(params, H, h)
    u = orc["F"][k] @ z + orc["f"][k]
    m = A @ z + B*u; d = C @ z + D*u + Sg
    Pn, sn = orc["Pval"][k+1], orc["s"][k+1]
    p_cur = (orc["Pval"][k] @ z + orc["s"][k])[0]
    p_nxt = (Pn @ m + sn)[0]
    q = (Pn @ d)[0]
    Pi = G[k][0, 0]
    sigma_star = d[0]; sigma_bar = sigma_star - gu*u
    zeta = q - Pi*sigma_star
    u_rec = -(b*p_nxt + gu*zeta + gu*Pi*sigma_bar)/(R + gu*gu*Pi)
    return dict(u=u, p_cur=p_cur, p_nxt=p_nxt, q=q, Pi=Pi, zeta=zeta,
                u_rec=u_rec, sigma_star=sigma_star, sigma_bar=sigma_bar)

def V(z, P, s, c): return 0.5*z @ P @ z + s @ z + c

# ---------------------------------------------------------------- checks
def run_checks():
    from scipy.optimize import minimize_scalar
    rng = np.random.default_rng(0)
    params = dict(a=-0.3, ad=0.6, b=1.0, s0=0.2, cx=0.1, cy=0.15, gu=0.4,
                  Q=1.0, R=0.1, QT=2.0)
    T, delta, h = 1.0, 0.2, 0.05
    N, H = round(T/h), round(delta/h)
    xref = 0.5*np.sin(2*np.pi*np.arange(N)*h/T); xtar = 0.3
    Q, R, QT = params["Q"], params["R"], params["QT"]
    b, s0, gu = params["b"], params["s0"], params["gu"]

    orc = riccati(params, H, h, N, xref, xtar)
    G = detached_curvature(params, H, h, N)
    A, B, C, D, Sg = build_dense(params, H, h); n = H+1
    e1 = np.zeros(n); e1[0] = 1.0

    # 0. structured vs dense recursion agreement
    P = QT*np.outer(e1, e1); err_struct = 0.0
    Pd = [None]*(N+1); Pd[N] = P
    for k in range(N-1, -1, -1):
        Lam = h*R + B@P@B + h*(D@P@D); K = B@P@A + h*(D@P@C)
        Pn = h*Q*np.outer(e1, e1) + A.T@P@A + h*(C.T@P@C) - np.outer(K, K)/Lam
        P = 0.5*(Pn+Pn.T); Pd[k] = P
        err_struct = max(err_struct, np.abs(P - orc["Pval"][k]).max())
    Gd = QT*np.outer(e1, e1)
    for k in range(N-1, -1, -1):
        Gd = h*Q*np.outer(e1, e1) + A.T@Gd@A + h*(C.T@Gd@C)
        err_struct = max(err_struct, np.abs(0.5*(Gd+Gd.T) - G[k]).max())
    print("structured-vs-dense max diff        =", err_struct)

    # 1. Bellman residual AT the analytic action (two-point exact quadrature)
    #    + action-location check via bounded scalar minimisation
    err_V, err_u = 0.0, 0.0
    for _ in range(300):
        k = rng.integers(0, N); z = rng.normal(0, 1.0, n)
        Pn, sn, cn = orc["Pval"][k+1], orc["s"][k+1], orc["c"][k+1]
        def rhs(u):
            m = A @ z + B*u; d = C @ z + D*u + Sg
            run = h*(0.5*Q*(z[0]-xref[k])**2 + 0.5*R*u*u)
            return run + 0.5*(V(m+np.sqrt(h)*d, Pn, sn, cn) + V(m-np.sqrt(h)*d, Pn, sn, cn))
        u_star = orc["F"][k] @ z + orc["f"][k]
        assert abs(u_star) < 40.0
        err_V = max(err_V, abs(rhs(u_star) - V(z, orc["Pval"][k], orc["s"][k], orc["c"][k])))
        res = minimize_scalar(rhs, bounds=(u_star-10, u_star+10), method="bounded",
                              options=dict(xatol=1e-12))
        err_u = max(err_u, abs(res.x - u_star))
    print("Bellman residual at analytic u*     =", err_V)
    print("scalar-minimiser action gap         =", err_u)

    # 2. exact discrete q-form FOC + algebraic anchored-coordinate identity
    #    + recovery-input oracle self-consistency (u_rec == u*)
    err_foc, err_anc, err_rec, max_zeta, gap_pp = 0.0, 0.0, 0.0, 0.0, []
    for _ in range(300):
        k = rng.integers(0, N); z = rng.normal(0, 1.0, n)
        t = exact_recovery_inputs(k, z, params, H, h, orc, G, xref)
        u, q, Pi = t["u"], t["q"], t["Pi"]
        err_foc = max(err_foc, abs(R*u + b*t["p_nxt"] + gu*q))
        err_rec = max(err_rec, abs(t["u_rec"] - u))
        max_zeta = max(max_zeta, abs(t["zeta"]))
        gap_pp.append((t["p_nxt"] - t["p_cur"], t["p_cur"]))
        sigma_bar = (C @ z + Sg)[0]
        for Pi_arb in (0.0, 0.37, 2.5):        # identity holds for ARBITRARY Pi
            zt = q - Pi_arb*(sigma_bar + gu*u)
            err_anc = max(err_anc, abs((R+gu*gu*Pi_arb)*u + (b*t["p_nxt"] + gu*zt + gu*Pi_arb*sigma_bar)))
    print("q-form FOC max residual             =", err_foc)
    print("anchored algebraic-identity max res =", err_anc)
    print("recovery-input oracle max |u_rec-u*|=", err_rec, " max|zeta| =", max_zeta)

    # 3. Monte Carlo check of the detached Hessian: average of the exact
    #    pathwise recursion  P~_k = hQ e1e1' + (A+C dW)' P~_{k+1} (A+C dW)
    Nm, Hm, M = 10, 2, 200_000
    pm = dict(params); nm = Hm+1
    Am, Bm, Cm, Dm, Sgm = build_dense(pm, Hm, h)
    e1m = np.zeros(nm); e1m[0] = 1.0
    Gm = detached_curvature(pm, Hm, h, Nm)
    dW = rng.normal(0, np.sqrt(h), (M, Nm))
    Pt = np.broadcast_to(QT*np.outer(e1m, e1m), (M, nm, nm)).copy()
    for k in range(Nm-1, -1, -1):
        J = Am[None, :, :] + dW[:, k, None, None]*Cm[None, :, :]
        Pt = h*Q*np.outer(e1m, e1m)[None] + np.einsum('mji,mjk,mkl->mil', J, Pt, J)
    mc_err = np.abs(Pt.mean(axis=0) - Gm[0]).max()
    mc_se = Pt.std(axis=0).max()/np.sqrt(M)
    print("detached-Hessian MC check |mean-G0| =", mc_err, " (max SE ~", mc_se, ")")

    # 4. headline numbers: Pval_11 vs Pi, and zeta at (k=0, z=0)
    gp = np.array(gap_pp)
    print("finite-h p alignment nRMSE(p_nxt, p_cur) =",
          f"{np.sqrt(np.mean(gp[:,0]**2))/np.sqrt(np.mean(gp[:,1]**2)):.4%}")
    t0 = exact_recovery_inputs(0, np.zeros(n), params, H, h, orc, G, xref)
    print(f"(Pval_0)_11 = {orc['Pval'][0][0,0]:.5f}   Pi_0 = {t0['Pi']:.5f}")
    print(f"k=0,z=0: u* = {t0['u']:.5f}  q = {t0['q']:.5f}  sigma* = {t0['sigma_star']:.5f}  zeta = {t0['zeta']:.5f}")
    assert err_struct < 1e-12 and err_foc < 1e-12 and err_anc < 1e-12
    assert err_rec < 1e-12 and err_V < 1e-8 and mc_err < 6*mc_se
    print("min Lam/h =", min(l/h for l in orc["Lam"]),
          "  min Pi_k =", min(G[k][0,0] for k in range(N+1)))


if __name__ == "__main__":
    run_checks()
