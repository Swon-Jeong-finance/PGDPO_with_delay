"""P3 budget-parameterized certainty-equivalent NMPC (CE-NMPC) PROXY
(baseline, NOT a reference/oracle; naming per review 2026-08-07 sec.5.3).

Receding-horizon CERTAINTY-EQUIVALENCE controller: at each step it rolls the
NOISE-FREE dynamics over `lookahead` steps from the current (lifted or
buffered) state, minimises the truncated cost over u in [0,1]^L with L-BFGS-B
under an explicit iteration budget, applies the first control, and warm-starts
the next solve with the shifted solution. The controlled diffusion
sigma0(1-eta u)I dW does NOT enter the surrogate objective or gradient --
that is the definition of the CE approximation, quantified on P3-R by the
paired dJ(CE-NMPC - HJB) cross-check. Budget status (sec.5.4): the knobs
(lookahead_steps, max_iter, gtol, ftol) are frozen in YAML and optimizer
health, deterministic-subproblem KKT, raw/deployed feasibility, and runtime
are recorded per controller.  Optimizer-only and full decision runtimes, plus
optimizer and total objective/gradient evaluation counts, are kept distinct.
CE-NMPC is the retained P3-D proxy; a scenario
variant is only a contingency if this proxy later proves clearly inadequate.

Two surrogate models share one adjoint interface:
  * kind="renewal":     state (I, M), dM = rho (I - M) dt -- EXACTLY the P3-R
                        lift, so the controller can be cross-validated against
                        the fine-grid HJB reference (paired dJ >= 0 up to MC
                        noise). This is the only way to certify the proxy
                        before deploying it on P3-D where no truth exists.
  * kind="distributed": state = I-buffer (H+1,), truncated-Gamma weights --
                        the P3-D deployment target.

Gradients are ANALYTIC reverse-mode adjoints of the deterministic recursion
(exact within the surrogate up to the full-truncation subgradient at I = 0),
verified against finite differences in the fast gate.  Before a prediction
window reaches physical T, (c_T/2) I_end^2 is an MPC terminal SURROGATE; when
the window reaches T it is the physical terminal cost.  No terminal-ablation
claim is made.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import minimize
from . import dynamics


def projected_box_kkt(u, grad, lower=0.0, upper=1.0):
    """Projected-gradient KKT diagnostic for a box-constrained minimisation.

    This certifies only the deterministic CE open-loop subproblem.  It is not
    the stochastic generalized-Hamiltonian KKT used later to compare PGDPO
    and learned baselines.
    """
    u, grad = np.asarray(u, dtype=float), np.asarray(grad, dtype=float)
    mapping = u - np.clip(u - grad, lower, upper)
    return dict(vector=mapping,
                inf=float(np.max(np.abs(mapping))),
                first=float(abs(mapping.flat[0])))


def _box_violation(x, lower=0.0, upper=1.0):
    x = np.asarray(x, dtype=float)
    return float(max(0.0, np.max(lower - x), np.max(x - upper)))


@dataclass(frozen=True)
class NMPCSolve:
    plan_raw: np.ndarray
    action_raw: float
    action_deployed: float
    objective: float
    kkt_inf: float
    kkt_first: float
    raw_plan_bound_violation: float
    raw_first_bound_violation: float
    deployment_clip_correction: float
    deployed_bound_violation: float
    success: bool
    status: int
    nit: int
    optimizer_nfev: int
    total_objective_grad_evals: int
    horizon: int
    optimizer_runtime_s: float
    decision_runtime_s: float


def _drift(p, Ib, M, u):
    return p["beta"]*(1.0 - Ib/p["Npop"])*M - p["gamma"]*Ib - p["b"]*u*Ib


def rollout_cost_grad_renewal(cfg, I0, M0, u):
    """Deterministic (I, M)-lift rollout: cost and exact d cost / d u."""
    p, dt = cfg["params"], cfg["dt"]
    L = len(u)
    I = np.empty(L+1); M = np.empty(L+1); I[0], M[0] = I0, M0
    cost = 0.0
    for j in range(L):
        Ib = max(I[j], 0.0)
        cost += dt*(0.5*p["c_I"]*Ib**2 + 0.5*p["R"]*u[j]**2)
        I[j+1] = I[j] + dt*_drift(p, Ib, M[j], u[j])
        M[j+1] = M[j] + dt*p["rho"]*(Ib - M[j])
    Ibe = max(I[L], 0.0)
    cost += 0.5*p["c_T"]*Ibe**2
    aI, aM = p["c_T"]*Ibe, 0.0                     # adjoint at the window end
    g = np.empty(L)
    for j in range(L-1, -1, -1):
        Ib = max(I[j], 0.0); s = 1.0 if I[j] > 0.0 else 0.0
        g[j] = dt*(p["R"]*u[j] - aI*p["b"]*Ib)
        dfdI = s*(-p["beta"]*M[j]/p["Npop"] - p["gamma"] - p["b"]*u[j])
        aI_new = aI*(1.0 + dt*dfdI) + aM*dt*p["rho"]*s + dt*p["c_I"]*Ib*s
        aM_new = aI*dt*p["beta"]*(1.0 - Ib/p["Npop"]) + aM*(1.0 - dt*p["rho"])
        aI, aM = aI_new, aM_new
    return cost, g


def rollout_cost_grad_dist(cfg, B0, w, u):
    """Deterministic buffered rollout: cost and exact d cost / d u.
    B0: (H+1,) with B0[0] = I_k; buffer shifts each step as in the shared
    simulator (full-truncation coefficients, M = max(B,0) @ w)."""
    p, dt = cfg["params"], cfg["dt"]
    L, n = len(u), len(B0)
    Bs = np.empty((L+1, n)); Bs[0] = B0
    cost = 0.0
    for j in range(L):
        B = Bs[j]; Ib = max(B[0], 0.0)
        M = np.maximum(B, 0.0) @ w
        cost += dt*(0.5*p["c_I"]*Ib**2 + 0.5*p["R"]*u[j]**2)
        Bs[j+1, 0] = B[0] + dt*_drift(p, Ib, M, u[j])
        Bs[j+1, 1:] = B[:-1]
    Ibe = max(Bs[L, 0], 0.0)
    cost += 0.5*p["c_T"]*Ibe**2
    a = np.zeros(n); a[0] = p["c_T"]*Ibe           # adjoint on the end buffer
    g = np.empty(L)
    for j in range(L-1, -1, -1):
        B = Bs[j]; Ib = max(B[0], 0.0)
        M = np.maximum(B, 0.0) @ w
        sB = (B > 0.0).astype(float)               # truncation subgradients
        g[j] = dt*(p["R"]*u[j] - a[0]*p["b"]*Ib)
        coefM = p["beta"]*(1.0 - Ib/p["Npop"])     # dI_next/dB via the kernel
        dfdI0 = sB[0]*(-p["beta"]*M/p["Npop"] - p["gamma"] - p["b"]*u[j])
        a_new = np.empty(n)
        # shift channel: B_next[i+1] = B[i]  ->  a_new[i] += a[i+1] (i < n-1)
        a_new[:-1] = a[1:]; a_new[-1] = 0.0
        # I_next channel through M (kernel taps, incl. tap 0) ...
        a_new += a[0]*dt*coefM*w*sB
        # ... and through the identity + direct-Ib part at tap 0, plus the
        # running-cost derivative
        a_new[0] += a[0]*(1.0 + dt*dfdI0) + dt*p["c_I"]*Ib*sB[0]
        a = a_new
    return cost, g


class NMPCController:
    """Stateful per-path receding-horizon controller with warm starts.
    kind = "renewal" | "distributed"."""
    def __init__(self, cfg, kind, lookahead=None, max_iter=None):
        if kind not in ("renewal", "distributed"):
            raise ValueError("kind must be renewal or distributed")
        self.cfg, self.kind = cfg, kind
        nm = cfg["nmpc"]
        self.L0 = int(nm["lookahead_steps"] if lookahead is None else lookahead)
        self.max_iter = int(nm["max_iter"] if max_iter is None else max_iter)
        self.gtol, self.ftol = float(nm["gtol"]), float(nm["ftol"])
        self.terminal_mode = nm["terminal_mode"]
        if not (1 <= self.L0 <= cfg["N"] and self.max_iter >= 1):
            raise ValueError("invalid CE-NMPC lookahead/max_iter")
        self.w = dynamics.kernel_weights(cfg) if kind == "distributed" else None
        self.reset()

    def reset(self):
        self._warm = {}
        self.stats = dict(n_solves=0, nit_sum=0, optimizer_nfev_sum=0,
                          total_objective_grad_evals_sum=0, failures=0,
                          maxiter_stops=0, nonfinite_count=0)
        self._optimizer_runtime = []; self._decision_runtime = []
        self._horizon = []
        self._kkt = []; self._kkt_first = []
        self._raw_violation = []; self._raw_first_violation = []
        self._clip_correction = []; self._deployed_violation = []

    def solve(self, k, state, path_id=0):
        started = time.perf_counter()
        L = min(self.L0, self.cfg["N"] - k)
        u0 = self._warm.get(path_id)
        u0 = np.full(L, 0.5) if u0 is None else \
            np.clip(np.append(u0[1:], u0[-1])[:L], 0.0, 1.0)
        if self.kind == "renewal":
            I0, M0 = state
            fun = lambda u: rollout_cost_grad_renewal(self.cfg, I0, M0, u)
        else:
            fun = lambda u: rollout_cost_grad_dist(self.cfg, state, self.w, u)
        res = minimize(fun, u0, jac=True, method="L-BFGS-B",
                       bounds=[(0.0, 1.0)]*L,
                       options=dict(maxiter=self.max_iter, gtol=self.gtol,
                                    ftol=self.ftol))
        optimizer_runtime_s = time.perf_counter() - started
        optimizer_nfev = int(res.nfev)
        # scipy's nfev counts joint objective/gradient calls made by the
        # optimizer (jac=True).  The evaluator deliberately recomputes the
        # final objective and gradient once for the reported KKT diagnostic.
        total_objective_grad_evals = optimizer_nfev + 1
        self.stats["n_solves"] += 1
        self.stats["nit_sum"] += int(res.nit)
        self.stats["optimizer_nfev_sum"] += optimizer_nfev
        self.stats["total_objective_grad_evals_sum"] += \
            total_objective_grad_evals
        if not res.success:
            self.stats["failures"] += 1     # budget-exhausted solves are kept
        if int(res.status) == 1:
            self.stats["maxiter_stops"] += 1
        plan = np.asarray(res.x, dtype=float).copy()
        objective, grad = fun(plan)           # evaluator-side recomputation
        if not (np.isfinite(plan).all() and np.isfinite(grad).all()
                and np.isfinite(objective)):
            self.stats["nonfinite_count"] += 1
            raise FloatingPointError("CE-NMPC returned nonfinite plan/gradient")
        kkt = projected_box_kkt(plan, grad)
        action_raw = float(plan[0])
        action_deployed = float(np.clip(action_raw, 0.0, 1.0))
        raw_plan_bound_violation = _box_violation(plan)
        raw_first_bound_violation = _box_violation([action_raw])
        deployment_clip_correction = abs(action_deployed-action_raw)
        deployed_bound_violation = _box_violation([action_deployed])
        # Record all non-timing decision diagnostics before stopping the
        # paper-facing clock.  Thus decision time covers the final evaluator
        # call, projected KKT, clipping, warm-start update, and telemetry.
        self._warm[path_id] = plan
        self._horizon.append(L)
        self._kkt.append(kkt["inf"]); self._kkt_first.append(kkt["first"])
        self._raw_violation.append(raw_plan_bound_violation)
        self._raw_first_violation.append(raw_first_bound_violation)
        self._clip_correction.append(deployment_clip_correction)
        self._deployed_violation.append(deployed_bound_violation)
        decision_runtime_s = time.perf_counter() - started
        record = NMPCSolve(
            plan_raw=plan, action_raw=action_raw,
            action_deployed=action_deployed, objective=float(objective),
            kkt_inf=kkt["inf"], kkt_first=kkt["first"],
            raw_plan_bound_violation=raw_plan_bound_violation,
            raw_first_bound_violation=raw_first_bound_violation,
            deployment_clip_correction=deployment_clip_correction,
            deployed_bound_violation=deployed_bound_violation,
            success=bool(res.success), status=int(res.status),
            nit=int(res.nit), optimizer_nfev=optimizer_nfev,
            total_objective_grad_evals=total_objective_grad_evals,
            horizon=L, optimizer_runtime_s=float(optimizer_runtime_s),
            decision_runtime_s=float(decision_runtime_s))
        self._optimizer_runtime.append(record.optimizer_runtime_s)
        self._decision_runtime.append(record.decision_runtime_s)
        return record

    def action(self, k, state, path_id=0):
        return self.solve(k, state, path_id).action_deployed

    def budget_stats(self):
        """JSON-serialisable CE optimizer/KKT/constraint/budget contract."""
        n = max(1, self.stats["n_solves"])
        arr = lambda x: np.asarray(x, dtype=float)
        ort = arr(self._optimizer_runtime)
        drt = arr(self._decision_runtime)
        hz, kk = arr(self._horizon), arr(self._kkt)
        full_ort = ort[hz == self.L0] if len(ort) else np.array([])
        full_drt = drt[hz == self.L0] if len(drt) else np.array([])
        agg = lambda x, fn, default=0.0: float(fn(x)) if len(x) else default
        return dict(
            self.stats, lookahead_steps=self.L0, max_iter=self.max_iter,
            gtol=self.gtol, ftol=self.ftol, terminal_mode=self.terminal_mode,
            nit_mean=self.stats["nit_sum"]/n,
            optimizer_nfev_mean=self.stats["optimizer_nfev_sum"]/n,
            total_objective_grad_evals_mean=
                self.stats["total_objective_grad_evals_sum"]/n,
            success_rate=1.0 - self.stats["failures"]/n,
            maxiter_fraction=self.stats["maxiter_stops"]/n,
            optimizer_runtime_total_s=float(ort.sum()) if len(ort) else 0.0,
            optimizer_runtime_mean_ms=agg(ort, np.mean)*1e3,
            optimizer_runtime_median_ms=agg(ort, np.median)*1e3,
            optimizer_runtime_p95_ms=
                agg(ort, lambda x: np.quantile(x, .95))*1e3,
            full_horizon_optimizer_runtime_mean_ms=
                agg(full_ort, np.mean)*1e3,
            decision_runtime_total_s=float(drt.sum()) if len(drt) else 0.0,
            decision_runtime_mean_ms=agg(drt, np.mean)*1e3,
            decision_runtime_median_ms=agg(drt, np.median)*1e3,
            decision_runtime_p95_ms=
                agg(drt, lambda x: np.quantile(x, .95))*1e3,
            full_horizon_decision_runtime_mean_ms=
                agg(full_drt, np.mean)*1e3,
            ce_subproblem_kkt_inf_mean=agg(kk, np.mean),
            ce_subproblem_kkt_inf_q95=agg(kk, lambda x: np.quantile(x, .95)),
            ce_subproblem_kkt_inf_max=agg(kk, np.max),
            ce_first_action_kkt_max=agg(arr(self._kkt_first), np.max),
            raw_plan_bound_violation_max=agg(arr(self._raw_violation), np.max),
            raw_first_bound_violation_max=agg(arr(self._raw_first_violation), np.max),
            deployment_clip_correction_max=agg(arr(self._clip_correction), np.max),
            deployed_bound_violation_max=agg(arr(self._deployed_violation), np.max))


def nmpc_policy_renewal(cfg, lookahead=None, max_iter=None):
    """pol(k, I, M) vectorized over paths for dynamics.simulate_paired."""
    ctrl = NMPCController(cfg, "renewal", lookahead, max_iter)
    def pol(k, I, M):
        I, M = np.atleast_1d(I), np.atleast_1d(M)
        return np.array([ctrl.action(k, (I[i], M[i]), path_id=i)
                         for i in range(len(I))])
    pol.controller = ctrl
    return pol


def nmpc_policy_dist(cfg, lookahead=None, max_iter=None):
    """pol(k, B) with B (Np, H+1) for dynamics.simulate_dist_paired."""
    ctrl = NMPCController(cfg, "distributed", lookahead, max_iter)
    def pol(k, B):
        B = np.atleast_2d(B)
        return np.array([ctrl.action(k, B[i], path_id=i)
                         for i in range(B.shape[0])])
    pol.controller = ctrl
    return pol
