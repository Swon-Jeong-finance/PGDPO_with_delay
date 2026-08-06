"""P2 production oracle: per-mode structured generalized Riccati for the
distributed-delay network LQ (unconstrained, m = d, common orthonormal
eigenbasis, all coefficient matrices diagonal in that basis, identical scalar
memory weights at every node -- outside these conditions the mode
decomposition does NOT hold; see the eigencheck notes).

Memory quadrature (manuscript Appendix C convention, paper-code gate):
  M_k = sum_{j=0..H} w_j X_{k-j},  normalized trapezoidal weights
  w_j prop K(j h) h with endpoint halving, sum_j w_j = 1,
  K(a) = rho e^{-rho a} / (1 - e^{-rho delta}).
The simulator must use the identical weights.

Complexity: shift + rank-one structured ops give O(d N H^2 (1+r)) total,
versus O(N (d(H+1))^3) for the dense buffer recursion.
"""
import numpy as np
from p1_oracle import AtPA, AtPe1, Ats

P2_API_VERSION = "p2-v1-trapezoid-modes"

def kernel_weights(rho, delta, h, H):
    ages = np.arange(H + 1) * h
    w = rho*np.exp(-rho*ages)/(1 - np.exp(-rho*delta))*h
    w[[0, -1]] *= 0.5
    return w / w.sum()

def mode_spec(V, mats):
    """Spectra of all coefficient matrices in the common eigenbasis V.
    mats: dict with A, AM, B, Cs[r], CMs[r], Ds[r], Q, R, QT, sig0[r]."""
    ev = lambda M: np.diag(V.T @ M @ V)
    r = len(mats["Cs"])
    return dict(a=ev(mats["A"]), aM=ev(mats["AM"]), b=ev(mats["B"]),
                c=[ev(M) for M in mats["Cs"]], cM=[ev(M) for M in mats["CMs"]],
                dd=[ev(M) for M in mats["Ds"]], q=ev(mats["Q"]), r=ev(mats["R"]),
                qT=ev(mats["QT"]), sig=[V.T @ v for v in mats["sig0"]], nbm=r)

def mode_rows(spec, i, w, h, n1):
    e1 = np.zeros(n1); e1[0] = 1.0
    row = e1*(1 + h*spec["a"][i] + h*spec["aM"][i]*w[0]); row[1:] += h*spec["aM"][i]*w[1:]
    crows = []
    for l in range(spec["nbm"]):
        c = e1*(spec["c"][l][i] + spec["cM"][l][i]*w[0]); c[1:] += spec["cM"][l][i]*w[1:]
        crows.append(c)
    return row, crows

def p2_mode_oracle(spec, w, H, h, N):
    """Structured per-mode recursions. Returns per-mode dicts with
    Pval, s, c (value), F, f, Lam (feedback), G (detached curvature)."""
    n1 = H + 1; e1 = np.zeros(n1); e1[0] = 1.0
    out = []
    for i in range(len(spec["a"])):
        row, crows = mode_rows(spec, i, w, h, n1)
        bi = spec["b"][i]; Ri = spec["r"][i]; qi = spec["q"][i]
        ddi = [spec["dd"][l][i] for l in range(spec["nbm"])]
        sgi = [spec["sig"][l][i] for l in range(spec["nbm"])]
        ccT = sum(np.outer(c, c) for c in crows)
        P = spec["qT"][i]*np.outer(e1, e1); s = np.zeros(n1); cc = 0.0
        G = P.copy()
        Ps, ss, cs, Fs, fs, Ls, Gs = [P], [s], [cc], [], [], [], [G.copy()]
        for k in range(N-1, -1, -1):
            P11 = P[0, 0]
            Lam = h*Ri + (h*bi)**2*P11 + h*sum(d*d for d in ddi)*P11
            K = h*bi*AtPe1(P, row) + h*P11*sum(d*c for d, c in zip(ddi, crows))
            kap = h*bi*s[0] + h*P11*sum(d*sg for d, sg in zip(ddi, sgi))
            Pn = h*qi*np.outer(e1, e1) + AtPA(P, row) + h*P11*ccT - np.outer(K, K)/Lam
            sn = Ats(s, row) + h*P11*sum(sg*c for sg, c in zip(sgi, crows)) - K*(kap/Lam)
            cn = cc + 0.5*h*P11*sum(sg*sg for sg in sgi) - 0.5*kap*kap/Lam
            Fs.insert(0, -K/Lam); fs.insert(0, -kap/Lam); Ls.insert(0, Lam)
            P, s, cc = 0.5*(Pn + Pn.T), sn, cn
            Ps.insert(0, P); ss.insert(0, s); cs.insert(0, cc)
            G = h*qi*np.outer(e1, e1) + AtPA(G, row) + h*G[0, 0]*ccT
            G = 0.5*(G + G.T); Gs.insert(0, G.copy())
        out.append(dict(Pval=Ps, s=ss, c=cs, F=Fs, f=fs, Lam=Ls, G=Gs))
    return out

def exact_recovery_inputs_p2(i, k, zi, spec, orc_i, w, H, h):
    """Exact per-mode recovery targets under the optimal policy:
      u, p_cur, p_nxt, per-Brownian q_l, Pi, per-Brownian zeta_l,
      sigma_star_l, sigma_bar_l, and the anchored recovered action."""
    n1 = H + 1
    row, crows = mode_rows(spec, i, w, h, n1)
    bi, Ri = spec["b"][i], spec["r"][i]
    ddi = [spec["dd"][l][i] for l in range(spec["nbm"])]
    sgi = [spec["sig"][l][i] for l in range(spec["nbm"])]
    u = orc_i["F"][k] @ zi + orc_i["f"][k]
    m = np.empty(n1); m[0] = row @ zi + h*bi*u; m[1:] = zi[:-1]
    Pn, sn = orc_i["Pval"][k+1], orc_i["s"][k+1]
    p_cur = (orc_i["Pval"][k] @ zi + orc_i["s"][k])[0]
    p_nxt = (Pn @ m + sn)[0]
    Pi = orc_i["G"][k][0, 0]
    sig_star = [crows[l] @ zi + ddi[l]*u + sgi[l] for l in range(spec["nbm"])]
    sig_bar = [ss - ddi[l]*u for l, ss in enumerate(sig_star)]
    q = [Pn[0, 0]*ss for ss in sig_star]
    zeta = [q[l] - Pi*sig_star[l] for l in range(spec["nbm"])]
    num = bi*p_nxt + sum(ddi[l]*(zeta[l] + Pi*sig_bar[l]) for l in range(spec["nbm"]))
    u_rec = -num/(Ri + Pi*sum(d*d for d in ddi))
    return dict(u=u, p_cur=p_cur, p_nxt=p_nxt, q=q, Pi=Pi, zeta=zeta,
                sig_star=sig_star, sig_bar=sig_bar, u_rec=u_rec)
