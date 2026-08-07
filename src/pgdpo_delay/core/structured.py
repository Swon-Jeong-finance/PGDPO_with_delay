"""Shared shift + rank-one structured linear algebra for buffer-state
recursions (owned by core: problems must not cross-import each other).

Buffer transition A = E + e1 r^T (down-shift plus first row); rank-one noise
rows C = e1 c^T. These identities give O(n^2) Riccati/Lyapunov steps."""
import numpy as np

def AtPA(P, r):
    """A^T P A for A = E + e1 r^T, symmetric P.  O(n^2)."""
    out = np.zeros_like(P)
    out[:-1, :-1] = P[1:, 1:]                 # E^T P E
    v = P[1:, 0]                              # E^T P e1
    out[:-1, :] += np.outer(v, r)
    out[:, :-1] += np.outer(r, v)             # r e1^T P E  (symmetry)
    out += P[0, 0]*np.outer(r, r)
    return out

def Ats(s, r):
    out = np.zeros_like(s); out[:-1] = s[1:]; return out + r*s[0]

def AtPe1(P, r):
    v = np.zeros(len(r)); v[:-1] = P[1:, 0]; return v + r*P[0, 0]
