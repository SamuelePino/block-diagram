"""Tail-Norm k-means (TN-k-means): an exact accelerator for Lloyd's algorithm.

Idea
----
Fix any orthonormal d x M matrix Q (here: approximate top-M principal
directions estimated from a subsample).  For a point x and a centre c, after
subtracting the data mean,

    ||x - c||^2 = ||Q_m^T x - Q_m^T c||^2 + ||x_perp - c_perp||^2 ,

where Q_m holds the first m columns of Q and x_perp is the part of x outside
span(Q_m).  With the scalar "tail norms" t_x = ||x_perp||, t_c = ||c_perp||,
Cauchy-Schwarz gives a two-sided bound from an m-dimensional computation:

    P_m + (t_x - t_c)^2  <=  ||x - c||^2  <=  P_m + (t_x + t_c)^2 .

A centre whose lower bound exceeds the smallest upper bound cannot be the
nearest one.  Points left with one candidate are assigned without touching the
d-dimensional data; the others move to the next level (more coordinates), and
only the surviving (point, centre) pairs after the last level get an exact
d-dimensional distance.  The bounds hold for ANY orthonormal Q, so the result is
the same partition Lloyd produces; the quality of Q only affects speed.

The same pass yields, for free, an upper bound on the distance to the assigned
centre and a lower bound on the distance to every other centre, which are
exactly the two per-point bounds Hamerly's algorithm maintains across
iterations.  `hamerly=True` uses them to skip points entirely.

All variants (plain Lloyd, Hamerly, TN, TN+Hamerly) share this code path and
the same BLAS, so timings differ only by the pruning strategy.
"""
import ctypes
import os
import time
import numpy as np
from scipy import sparse


def top_basis(S, M, rng, iters=2):
    """Approximate top-M right singular vectors of S (rows = samples)."""
    d = S.shape[1]
    M = min(M, d)
    p = min(d, M + 8)
    Y = S.T @ (S @ rng.standard_normal((d, p)))
    for _ in range(iters):
        Qm, _ = np.linalg.qr(Y)
        Y = S.T @ (S @ Qm)
    Qm, _ = np.linalg.qr(Y)
    _, _, Vt = np.linalg.svd(S @ Qm, full_matrices=False)
    return Qm @ Vt.T[:, :M]


def _group_sums(X, rows, labels, k):
    """sums[j] = sum of X[rows] with labels == j   (rows=None -> all rows)."""
    n = len(labels)
    H = sparse.csr_matrix((np.ones(n), (labels, np.arange(n))), shape=(k, n))
    return H @ (X if rows is None else X[rows])


def _load_kernel():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libtnk.so')
    if not os.path.exists(path):
        return None
    lib = ctypes.CDLL(path)
    lib.tn_assign.restype = None
    return lib


_LIB = _load_kernel()
_P = ctypes.c_void_p


class TNKMeans:
    def __init__(self, levels=(16, 48, 128), hamerly=True, max_iter=300,
                 sample=4000, seed=0, pair_frac=0.25, chunk_elems=1 << 22,
                 kernel='c'):
        self.kernel = kernel if _LIB is not None else 'numpy'
        self.levels = tuple(levels)
        self.hamerly = hamerly
        self.max_iter = max_iter
        self.sample = sample
        self.seed = seed
        self.pair_frac = pair_frac
        self.chunk_elems = chunk_elems

    # ------------------------------------------------------------------ setup
    def _prepare(self, X):
        n, d = X.shape
        self.xn = np.einsum('ij,ij->i', X, X)
        if not self.levels:
            return
        rng = np.random.default_rng(self.seed)
        self.mu = X.mean(axis=0)
        idx = rng.choice(n, min(n, self.sample), replace=False)
        idx.sort()
        Q = top_basis(X[idx] - self.mu, self.levels[-1], rng)
        self.Q = Q
        self.lv = [m for m in self.levels if m <= Q.shape[1]]
        self.Z = X @ Q
        self.Z -= self.mu @ Q
        xc2 = self.xn - 2.0 * (X @ self.mu) + self.mu @ self.mu
        L = len(self.lv)
        self.zblk = np.empty((n, L))
        a = 0
        for l, b in enumerate(self.lv):
            self.zblk[:, l] = np.einsum('ij,ij->i', self.Z[:, a:b], self.Z[:, a:b])
            a = b
        self.xtail2 = np.maximum(xc2[:, None] - np.cumsum(self.zblk, axis=1), 0.0)
        self.xtail = np.sqrt(self.xtail2)

    def _prepare_centers(self, C):
        self.C = C
        self.cn = np.einsum('ij,ij->i', C, C)
        if not self.levels:
            return
        Cc = C - self.mu
        self.CZ = Cc @ self.Q
        cc2 = np.einsum('ij,ij->i', Cc, Cc)
        L = len(self.lv)
        self.cblk = np.empty((L, len(C)))
        a = 0
        for l, b in enumerate(self.lv):
            self.cblk[l] = np.einsum('ij,ij->i', self.CZ[:, a:b], self.CZ[:, a:b])
            a = b
        self.ctail2 = np.maximum(cc2[None, :] - np.cumsum(self.cblk, axis=0), 0.0)
        self.ctail = np.sqrt(self.ctail2)
        self.CZ = np.ascontiguousarray(self.CZ)

    # ------------------------------------------------------------- assignment
    def _assign(self, X, rows, prev=None, init=None):
        """Exact nearest centre for `rows` (index array or None for all).

        Returns labels, squared upper bound on the distance to the assigned
        centre, squared lower bound on the distance to any other centre.
        """
        n = X.shape[0] if rows is None else len(rows)
        k = len(self.C)
        lab = np.empty(n, dtype=np.int64)
        ub2 = np.empty(n)
        lb2 = np.empty(n)
        if self.kernel == 'c' and self.levels:
            L = len(self.lv)
            lev = np.array(self.lv, dtype=np.int64)
            stat = np.zeros(2 + L, dtype=np.int64)
            C = np.ascontiguousarray(self.C)
            CZT = np.ascontiguousarray(self.CZ[:, :self.lv[0]].T)
            rp = None if rows is None else np.ascontiguousarray(rows, dtype=np.int64)
            a = lambda v: None if v is None else _P(v.ctypes.data)
            L_ = ctypes.c_long
            _LIB.tn_assign(a(X), a(self.Z), a(self.xtail), a(self.xn), a(C), a(self.CZ), a(CZT),
                           a(self.ctail), a(self.cn), a(rp), L_(n), L_(X.shape[1]),
                           L_(self.Z.shape[1]), L_(k), L_(L), a(lev), a(prev), a(init),
                           a(lab), a(ub2), a(lb2), a(stat))
            self.stats['lowdim'] += int(stat[0])
            self.stats['pairs'] += int(stat[1])
            self.stats['decided'][:L] += stat[2:]
            return lab, ub2, lb2
        step = max(256, self.chunk_elems // k)
        for s in range(0, n, step):
            e = min(n, s + step)
            cur = np.arange(s, e) if rows is None else rows[s:e]
            sl = slice(s, e) if rows is None else None
            self._assign_chunk(X, cur, sl, lab[s:e], ub2[s:e], lb2[s:e])
        return lab, ub2, lb2

    def _sketch_ub2(self, rows, lab):
        """Squared upper bound on ||x - c_lab|| from the M-coordinate sketch."""
        n = len(lab)
        out = np.empty(n)
        if self.kernel == 'c':
            a = lambda v: None if v is None else _P(v.ctypes.data)
            L_ = ctypes.c_long
            lab = np.ascontiguousarray(lab, dtype=np.int64)
            rp = None if rows is None else np.ascontiguousarray(rows, dtype=np.int64)
            _LIB.tn_ub(a(self.Z), a(self.xtail), a(self.CZ), a(self.ctail), a(rp), L_(n),
                       L_(self.Z.shape[1]), L_(len(self.C)), L_(len(self.lv)), a(lab), a(out))
            return out
        Dz = (self.Z if rows is None else self.Z[rows]) - self.CZ[lab]
        out = np.einsum('ij,ij->i', Dz, Dz)
        xt = self.xtail[:, -1] if rows is None else self.xtail[rows, -1]
        out += (xt + self.ctail[-1][lab]) ** 2
        return out

    def _assign_chunk(self, X, cur, sl, lab, ub2, lb2):
        k = len(self.C)
        st = self.stats
        pos = np.arange(len(cur))          # positions inside the output views
        P = None
        LB = alive = None
        a = 0
        if self.levels:
            for l, b in enumerate(self.lv):
                Zb = self.Z[sl, a:b] if (sl is not None and P is None) else self.Z[cur, a:b]
                inc = Zb @ self.CZ[:, a:b].T
                inc *= -2.0
                inc += self.zblk[cur, l][:, None]
                inc += self.cblk[l][None, :]
                P = inc if P is None else P + inc
                st['lowdim'] += len(cur) * k * (b - a)
                a = b
                # everything below is relative to the row constant t_x^2
                G = P + self.ctail2[l][None, :]
                W = (2.0 * self.xtail[cur, l])[:, None] * self.ctail[l][None, :]
                UB = G + W
                LB = G
                LB -= W
                best = UB.argmin(axis=1)
                ar = np.arange(len(cur))
                minUB = UB[ar, best]
                tx2 = self.xtail2[cur, l]
                eps = 1e-10 * (minUB + tx2 + 1.0)
                alive = LB <= (minUB + eps)[:, None]
                dec = alive.sum(axis=1) == 1
                if dec.any():
                    p = pos[dec]
                    lab[p] = best[dec]
                    ub2[p] = minUB[dec] + tx2[dec]
                    Ld = LB[dec]
                    Ld[np.arange(len(p)), best[dec]] = np.inf
                    lb2[p] = Ld.min(axis=1) + tx2[dec]
                    st['decided'][l] += len(p)
                    und = ~dec
                    cur, pos, P = cur[und], pos[und], P[und]
                    LB, alive = LB[und], alive[und]
                    LB += self.xtail2[cur, l][:, None]
                else:
                    LB += tx2[:, None]
                sl = None
                if len(cur) == 0:
                    return
        m = len(cur)
        xn = self.xn[cur]
        if alive is not None and alive.sum() < self.pair_frac * alive.size:
            # exact distances only for surviving (point, centre) pairs
            pi, pj = np.nonzero(alive)
            E = LB                                   # true lower bounds so far
            ex = np.empty(len(pi))
            for s in range(0, len(pi), 8192):
                e = min(len(pi), s + 8192)
                ex[s:e] = np.einsum('ij,ij->i', X[cur[pi[s:e]]], self.C[pj[s:e]])
            ex *= -2.0
            ex += xn[pi] + self.cn[pj]
            E[pi, pj] = ex
            st['pairs'] += len(pi)
            # best is among alive pairs; pruned entries keep LB > min UB >= best
            Em = np.where(alive, E, np.inf)
            best = Em.argmin(axis=1)
            ar = np.arange(m)
            lab[pos] = best
            ub2[pos] = Em[ar, best]
            E[ar, best] = np.inf
            lb2[pos] = E.min(axis=1)
        else:
            D = X[cur] @ self.C.T
            D *= -2.0
            D += self.cn[None, :]
            st['pairs'] += m * k
            best = D.argmin(axis=1)
            ar = np.arange(m)
            lab[pos] = best
            ub2[pos] = D[ar, best] + xn
            D[ar, best] = np.inf
            lb2[pos] = D.min(axis=1) + xn

    # -------------------------------------------------------------------- fit
    def fit(self, X, C0):
        t0 = time.perf_counter()
        n, d = X.shape
        k = len(C0)
        L = len(self.levels)
        self.stats = {'lowdim': 0, 'pairs': 0, 'decided': np.zeros(max(L, 1), dtype=np.int64),
                      'skipped': 0, 'visited': 0}
        self._prepare(X)
        self.t_prep = time.perf_counter() - t0
        self._prepare_centers(np.array(C0, dtype=X.dtype))
        labels, ub2, lb2 = self._assign(X, None)
        self.stats['visited'] += n
        ub = np.sqrt(np.maximum(ub2, 0.0))
        lb = np.sqrt(np.maximum(lb2, 0.0))
        sums = _group_sums(X, None, labels, k)
        counts = np.bincount(labels, minlength=k).astype(np.float64)
        it = 0
        for it in range(1, self.max_iter + 1):
            C = self.C.copy()
            nz = counts > 0
            C[nz] = sums[nz] / counts[nz][:, None]
            shift = np.sqrt(np.einsum('ij,ij->i', C - self.C, C - self.C))
            self._prepare_centers(C)
            if self.hamerly:
                ub += shift[labels]
                o = np.argsort(shift)
                big, big2 = shift[o[-1]], (shift[o[-2]] if k > 1 else 0.0)
                lb -= np.where(labels == o[-1], big2, big)
                CC = self.cn[:, None] + self.cn[None, :] - 2.0 * (C @ C.T)
                np.fill_diagonal(CC, np.inf)
                sc = 0.5 * np.sqrt(np.maximum(CC.min(axis=1), 0.0))
                thr = np.maximum(lb, sc[labels])
                cand = np.nonzero(ub > thr)[0]
                prev = init = None
                if len(cand):                        # tighten the upper bound
                    la = labels[cand]
                    if self.levels:
                        t = self._sketch_ub2(cand, la)
                    else:
                        t = np.empty(len(cand))
                        for s in range(0, len(cand), 8192):
                            e = min(len(cand), s + 8192)
                            t[s:e] = np.einsum('ij,ij->i', X[cand[s:e]], C[la[s:e]])
                        t *= -2.0
                        t += self.xn[cand] + self.cn[la]
                    ub[cand] = np.sqrt(np.maximum(t, 0.0))
                    keep = ub[cand] > thr[cand]
                    cand = cand[keep]
                    if self.levels and self.kernel == 'c':
                        prev, init = np.ascontiguousarray(la[keep]), np.ascontiguousarray(t[keep])
                rows = cand
            else:
                rows = np.arange(n)
                prev = init = None
                if self.levels and self.kernel == 'c':
                    prev, init = labels, self._sketch_ub2(None, labels)
            self.stats['visited'] += len(rows)
            self.stats['skipped'] += n - len(rows)
            if len(rows) == 0:
                break
            nl, u2, l2 = self._assign(X, None if len(rows) == n else rows, prev, init)
            if self.hamerly:
                ub[rows] = np.sqrt(np.maximum(u2, 0.0))
                lb[rows] = np.sqrt(np.maximum(l2, 0.0))
            ch = nl != labels[rows]
            nch = int(ch.sum())
            if nch == 0:
                break
            if nch > n // 3:
                labels[rows] = nl
                sums = _group_sums(X, None, labels, k)
            else:
                cr = rows[ch]
                sums -= _group_sums(X, cr, labels[cr], k)
                sums += _group_sums(X, cr, nl[ch], k)
                labels[cr] = nl[ch]
            counts = np.bincount(labels, minlength=k).astype(np.float64)
        self.n_iter = it
        self.labels = labels
        self.centers = self.C
        self.time = time.perf_counter() - t0
        # sum_i ||x_i - c_{a(i)}||^2 from the per-cluster sums (no n x d temporary)
        sums = _group_sums(X, None, labels, k)
        counts = np.bincount(labels, minlength=k)
        self.inertia = float(self.xn.sum() - 2.0 * np.einsum('ij,ij->', sums, self.C)
                             + counts @ self.cn)
        return self
