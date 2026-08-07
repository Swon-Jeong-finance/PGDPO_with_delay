"""Estimator layer contract (torch port pending).

The production estimators (p^, Pi^ via detached fixed-control OL-BPTT;
zeta^ via anchored nested antithetic CRN regression) will live here as the
single autodiff implementation. The verified numpy prototypes remain inside
problems/p1 (they are P1-specific affine-buffer specialisations kept as the
numerical contract for the torch port; promoting them as-is would be fake
generality)."""
