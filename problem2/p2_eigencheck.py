"""P2 eigenmode decomposition correctness harness (post-review).

Builds the dense d(H+1)-dim buffer generalized Riccati (value P, affine s,
constant c, detached G) under the manuscript's NORMALIZED TRAPEZOIDAL memory
quadrature M_k = sum_{j=0..H} w_j X_{k-j}, and compares against the
production per-mode structured oracle (p2_oracle.p2_mode_oracle) to machine
precision, including:
  P, G, s assemblies; c_dense == sum_i c_mode; V' Lam V diagonality;
  feedback actions; vector q-form FOC  R u* + B'p + sum_l D_l' q_l = 0
  in BOTH coordinates; per-mode anchored recovery identity (u_rec == u*).
Decomposition scope: m=d, unconstrained, common orthonormal basis, all
coefficient matrices diagonal in it, identical scalar memory weights per node.
"""
import numpy as np
from p2_oracle import P2_API_VERSION, kernel_weights, mode_spec, p2_mode_oracle, \
                      exact_recovery_inputs_p2
assert P2_API_VERSION == "p2-v1-trapezoid-modes"

rng = np.random.default_rng(0)
d, r, H = 6, 2, 5
T, h = 1.0, 0.05
N = round(T/h); delta = H*h; nb = d*(H+1); n1 = H+1

S = np.roll(np.eye(d), 1, axis=1)
L = 2*np.eye(d) - S - S.T
lam, V = np.linalg.eigh(L)
f = lambda a0, a1: a0*np.eye(d) + a1*L
mats = dict(A=f(-0.5, -0.3), AM=f(0.4, 0.1), B=f(1.0, 0.2),
            Cs=[f(0.15, 0.05), f(0.10, -0.03)], CMs=[f(0.10, 0.0), f(0.0, 0.05)],
            Ds=[f(0.30, 0.10), f(0.15, 0.0)],
            sig0=[rng.normal(0, 0.2, d) for _ in range(r)],
            Q=f(1.0, 0.2), R=f(0.1, 0.02), QT=f(2.0, 0.0))
rho = 5.0
w = kernel_weights(rho, delta, h, H)          # trapezoidal, j = 0..H, sum = 1

# ---------------- dense buffer reference (tap-major) ------------------------
def blockmat(first_blocks):
    M = np.zeros((nb, nb))
    for j, Bj in enumerate(first_blocks): M[0:d, j*d:(j+1)*d] = Bj
    return M
AZ = blockmat([np.eye(d) + h*mats["A"] + h*mats["AM"]*w[0]]
              + [h*mats["AM"]*w[j] for j in range(1, H+1)])
for j in range(1, H+1): AZ[j*d:(j+1)*d, (j-1)*d:j*d] = np.eye(d)
CZ = [blockmat([mats["Cs"][l] + mats["CMs"][l]*w[0]]
               + [mats["CMs"][l]*w[j] for j in range(1, H+1)]) for l in range(r)]
BZ = np.zeros((nb, d)); BZ[0:d] = h*mats["B"]
DZ = [np.zeros((nb, d)) for _ in range(r)]
for l in range(r): DZ[l][0:d] = mats["Ds"][l]
SZ = [np.zeros(nb) for _ in range(r)]
for l in range(r): SZ[l][0:d] = mats["sig0"][l]
QZ = np.zeros((nb, nb)); QZ[0:d, 0:d] = mats["Q"]

P = np.zeros((nb, nb)); P[0:d, 0:d] = mats["QT"]; s = np.zeros(nb); c = 0.0
Pd, sd, cd, Fd, fd, Ld = [P], [s], [c], [], [], []
Gm = P.copy(); Gd = [Gm.copy()]
for k in range(N-1, -1, -1):
    Lam = h*mats["R"] + BZ.T @ P @ BZ + h*sum(DZ[l].T @ P @ DZ[l] for l in range(r))
    K   = BZ.T @ P @ AZ + h*sum(DZ[l].T @ P @ CZ[l] for l in range(r))
    kap = BZ.T @ s + h*sum(DZ[l].T @ P @ SZ[l] for l in range(r))
    Li = np.linalg.inv(Lam)
    Pn = QZ*h + AZ.T @ P @ AZ + h*sum(CZ[l].T @ P @ CZ[l] for l in range(r)) - K.T @ Li @ K
    sn = AZ.T @ s + h*sum(CZ[l].T @ P @ SZ[l] for l in range(r)) - K.T @ (Li @ kap)
    cn = c + 0.5*h*sum(SZ[l] @ P @ SZ[l] for l in range(r)) - 0.5*kap @ Li @ kap
    Fd.insert(0, -Li @ K); fd.insert(0, -Li @ kap); Ld.insert(0, Lam)
    P, s, c = 0.5*(Pn + Pn.T), sn, cn
    Pd.insert(0, P); sd.insert(0, s); cd.insert(0, c)
    Gm = QZ*h + AZ.T @ Gm @ AZ + h*sum(CZ[l].T @ Gm @ CZ[l] for l in range(r))
    Gm = 0.5*(Gm + Gm.T); Gd.insert(0, Gm.copy())

# ---------------- production per-mode oracle --------------------------------
spec = mode_spec(V, mats)
orc = p2_mode_oracle(spec, w, H, h, N)

Ttap = np.kron(np.eye(n1), V.T)
def assemble(Ms):
    Mb = np.zeros((nb, nb))
    for i in range(d):
        for j in range(n1):
            Mb[j*d+i, np.arange(n1)*d+i] = Ms[i][j]
    return Ttap.T @ Mb @ Ttap
def assemble_vec(vs):
    v = np.zeros(nb)
    for i in range(d):
        v[np.arange(n1)*d+i] = vs[i]
    return Ttap.T @ v

eP = max(np.abs(Pd[k] - assemble([orc[i]["Pval"][k] for i in range(d)])).max() for k in range(N+1))
eG = max(np.abs(Gd[k] - assemble([orc[i]["G"][k] for i in range(d)])).max() for k in range(N+1))
es = max(np.abs(sd[k] - assemble_vec([orc[i]["s"][k] for i in range(d)])).max() for k in range(N+1))
ec = max(abs(cd[k] - sum(orc[i]["c"][k] for i in range(d))) for k in range(N+1))
eL = max(np.abs(V.T @ Ld[k] @ V - np.diag([orc[i]["Lam"][k] for i in range(d)])).max() for k in range(N))
eu = efoc_d = efoc_m = eanc = 0.0
for _ in range(200):
    k = rng.integers(0, N); z = rng.normal(0, 1, nb)
    u_dense = Fd[k] @ z + fd[k]
    zt = Ttap @ z
    tms = [exact_recovery_inputs_p2(i, k, zt[i::d], spec, orc[i], w, H, h) for i in range(d)]
    u_modes = V @ np.array([t["u"] for t in tms])
    eu = max(eu, np.abs(u_dense - u_modes).max())
    # per-mode q-form FOC and anchored recovery identity
    for i, t in enumerate(tms):
        efoc_m = max(efoc_m, abs(spec["r"][i]*t["u"] + spec["b"][i]*t["p_nxt"]
                     + sum(spec["dd"][l][i]*t["q"][l] for l in range(r))))
        eanc = max(eanc, abs(t["u_rec"] - t["u"]))
    # dense vector q-form FOC:  R u* + B'p + sum_l D_l' q_l = 0
    p_dense = V @ np.array([t["p_nxt"] for t in tms])
    q_dense = [V @ np.array([t["q"][l] for t in tms]) for l in range(r)]
    res = mats["R"] @ u_dense + mats["B"].T @ p_dense \
          + sum(mats["Ds"][l].T @ q_dense[l] for l in range(r))
    efoc_d = max(efoc_d, np.abs(res).max())
print(f"[{P2_API_VERSION}] d={d} r={r} H={H} N={N}  trapezoidal weights sum={w.sum():.3f}")
print("max|P_dense - assemble(P_modes)|   =", eP)
print("max|G_dense - assemble(G_modes)|   =", eG)
print("max|s_dense - assemble(s_modes)|   =", es)
print("max|c_dense - sum(c_modes)|        =", ec)
print("max|V'Lam V - diag(Lam_i)|         =", eL)
print("max|u_dense - V u_modes| (200 pts) =", eu)
print("mode q-form FOC max residual       =", efoc_m)
print("dense vector FOC max residual      =", efoc_d)
print("anchored recovery max |u_rec-u*|   =", eanc)
print("min Lam_i/h =", min(min(o["Lam"])/h for o in orc),
      " min Pi_i =", min(o["G"][k][0,0] for o in orc for k in range(N+1)))
