"""Benchmark TN-k-means against Lloyd / Hamerly (same code path) and scikit-learn.

usage: python bench.py DATA_DIR [dataset ...]
"""
import gzip
import sys
import time
import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus
from tnkm import TNKMeans


def load(name, ddir):
    rng = np.random.default_rng(1)
    if name == 'mnist':
        return np.load(f'{ddir}/mnist.npz')['x_train'].reshape(60000, -1).astype(np.float64)
    if name == 'fashion':
        with gzip.open(f'{ddir}/fashion.gz') as f:
            return np.frombuffer(f.read(), np.uint8, offset=16).reshape(60000, -1).astype(np.float64)
    if name == 'gmm':        # clustered data, decaying spectrum
        n, d, kk = 100000, 256, 200
        scale = 1.0 / np.sqrt(1.0 + np.arange(d))
        cen = rng.standard_normal((kk, d)) * scale * 3
        return cen[rng.integers(kk, size=n)] + rng.standard_normal((n, d)) * scale
    if name == 'flat':       # adversarial: isotropic Gaussian, no structure
        return rng.standard_normal((50000, 256))
    raise ValueError(name)


KS = (50, 200, 1000)


def main():
    ddir = sys.argv[1]
    names = sys.argv[2:] or ['mnist', 'fashion', 'gmm', 'flat']
    print('dataset      k  method        time(s)  speedup  iters  inertia_rel  agree   visited  exact_pair_frac  sketch_coords/pair')
    for name in names:
        X = load(name, ddir)
        n, d = X.shape
        for k in KS:
            C0, _ = kmeans_plusplus(X[:20000], k, random_state=0)
            variants = [
                ('lloyd', dict(levels=(), hamerly=False)),
                ('hamerly', dict(levels=(), hamerly=True)),
                ('tn', dict(hamerly=False)),
                ('tn+hamerly', dict(hamerly=True)),
            ]
            ref = None
            for vn, kw in variants:
                m = TNKMeans(max_iter=100, **kw).fit(X, C0)
                if ref is None:
                    ref = m
                st = m.stats
                print(f'{name:8s} {k:5d}  {vn:12s} {m.time:8.2f} {ref.time / m.time:7.2f}x {m.n_iter:6d} '
                      f'{m.inertia / ref.inertia - 1:+.1e}  {np.mean(m.labels == ref.labels):.4f} '
                      f'{st["visited"] / (n * (m.n_iter + 1)):8.3f} {st["pairs"] / (n * k * (m.n_iter + 1)):8.4f} {st["lowdim"] / (n * k * (m.n_iter + 1)):6.1f}',
                      flush=True)
            for alg in ('lloyd', 'elkan'):
                t = time.perf_counter()
                sk = KMeans(k, init=C0, n_init=1, max_iter=100, tol=0, algorithm=alg).fit(X)
                t = time.perf_counter() - t
                print(f'{name:8s} {k:5d}  sk-{alg:9s} {t:8.2f} {ref.time / t:7.2f}x {sk.n_iter_:6d} '
                      f'{sk.inertia_ / ref.inertia - 1:+.1e}  {np.mean(sk.labels_ == ref.labels):.4f}', flush=True)


if __name__ == '__main__':
    main()
