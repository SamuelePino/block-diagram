/* Tail-norm filtered nearest-centre assignment (see tnkm.py for the maths).
 * build: gcc -O3 -march=native -ffast-math -fopenmp -shared -fPIC tnk.c -o libtnk.so
 */
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <omp.h>

/* Upper bound (squared) on ||x_i - c_{lab[r]}|| from the full M-coordinate
 * sketch: P_M + (t_x + t_c)^2, tails taken at the last level. */
void tn_ub(const double *Z, const double *xt, const double *CZ, const double *ctT,
           const long *rows, long nr, long M, long k, long L, const long *lab, double *ub2)
{
    #pragma omp parallel for schedule(static)
    for (long r = 0; r < nr; r++) {
        long i = rows ? rows[r] : r, j = lab[r];
        const double *z = Z + i * M, *cz = CZ + j * M;
        double acc = 0.0;
        for (long t = 0; t < M; t++) { double df = z[t] - cz[t]; acc += df * df; }
        double hi = xt[i * L + L - 1] + ctT[(L - 1) * k + j];
        ub2[r] = acc + hi * hi;
    }
}

/* rows == NULL -> points 0..nr-1.  lev[l] = number of sketch coords used up
 * to level l (increasing, lev[L-1] <= M).  xt: n x L tail norms; ctT: L x k
 * centre tail norms; CZT: the first lev[0] sketch coords of the centres in
 * dimension-major layout (lev[0] x k) so level 0 vectorises across centres.
 * prev/init (optional): a centre and a valid squared upper bound on its
 * distance, used as the starting pruning threshold.
 * stat[0] += sketch coordinate operations, stat[1] += exact pairs,
 * stat[2+l] += points decided at level l. */
void tn_assign(const double *X, const double *Z, const double *xt, const double *xn,
               const double *C, const double *CZ, const double *CZT, const double *ctT,
               const double *cn, const long *rows, long nr, long d, long M, long k, long L,
               const long *lev, const long *prev, const double *init,
               long *lab, double *ub2, double *lb2, long *stat)
{
    long s_low = 0, s_pair = 0;
    #pragma omp parallel reduction(+:s_low, s_pair)
    {
        double *P = malloc(sizeof(double) * k);
        double *LBv = malloc(sizeof(double) * k);
        double *UBv = malloc(sizeof(double) * k);
        long *cand = malloc(sizeof(long) * k);
        long *dec = calloc(L + 1, sizeof(long));
        #pragma omp for schedule(dynamic, 256)
        for (long r = 0; r < nr; r++) {
            long i = rows ? rows[r] : r;
            const double *z = Z + i * M;
            long nc = 0, m0 = lev[0], done = 0;
            double pruned = DBL_MAX;      /* min lower bound over discarded centres */
            double thr0 = init ? init[r] : DBL_MAX;   /* best known upper bound */
            /* ---- level 0: all centres, vectorised across centres ---- */
            for (long j = 0; j < k; j++) P[j] = 0.0;
            for (long t = 0; t < m0; t++) {
                const double zt = z[t];
                const double *row = CZT + t * k;
                for (long j = 0; j < k; j++) { double df = zt - row[j]; P[j] += df * df; }
            }
            {
                double tx = xt[i * L], mu = thr0;
                for (long j = 0; j < k; j++) {
                    double tc = ctT[j], lo = tx - tc, hi = tx + tc;
                    LBv[j] = P[j] + lo * lo;
                    double u = P[j] + hi * hi;
                    UBv[j] = u;
                    mu = u < mu ? u : mu;
                }
                thr0 = mu;
                double thr = mu * (1.0 + 1e-10) + 1e-300;
                for (long j = 0; j < k; j++) {
                    if (LBv[j] <= thr) { cand[nc] = j; UBv[nc] = UBv[j]; nc++; }
                    else if (LBv[j] < pruned) pruned = LBv[j];
                }
                s_low += k * m0;
                if (nc == 1) { done = 1; dec[0]++; }
            }
            /* ---- finer levels: surviving centres only ---- */
            long a = m0;
            for (long l = 1; l < L && !done; l++) {
                long b = lev[l];
                double tx = xt[i * L + l], mu = thr0;
                const double *ctl = ctT + l * k;
                for (long q = 0; q < nc; q++) {
                    long j = cand[q];
                    const double *cz = CZ + j * M;
                    double acc = 0.0;
                    for (long t = a; t < b; t++) { double df = z[t] - cz[t]; acc += df * df; }
                    double p = P[j] + acc;
                    P[j] = p;
                    double tc = ctl[j], lo = tx - tc, hi = tx + tc;
                    LBv[q] = p + lo * lo;
                    double u = p + hi * hi;
                    UBv[q] = u;
                    mu = u < mu ? u : mu;
                }
                s_low += nc * (b - a);
                a = b;
                thr0 = mu;
                double thr = mu * (1.0 + 1e-10) + 1e-300;
                long m = 0;
                for (long q = 0; q < nc; q++) {
                    if (LBv[q] <= thr) { cand[m] = cand[q]; UBv[m] = UBv[q]; m++; }
                    else if (LBv[q] < pruned) pruned = LBv[q];
                }
                nc = m;
                if (nc == 1) { done = 1; dec[l]++; }
            }
            if (done) {
                double u = UBv[0];
                if (prev && prev[r] == cand[0] && init[r] < u) u = init[r];
                lab[r] = cand[0]; ub2[r] = u; lb2[r] = pruned;
                continue;
            }
            const double *x = X + i * d;
            double b1 = DBL_MAX, b2 = DBL_MAX;
            long best = -1;
            for (long q = 0; q < nc; q++) {
                long j = cand[q];
                const double *c = C + j * d;
                double dot = 0.0;
                for (long t = 0; t < d; t++) dot += x[t] * c[t];
                double e = xn[i] + cn[j] - 2.0 * dot;
                if (e < b1) { b2 = b1; b1 = e; best = j; }
                else if (e < b2) b2 = e;
            }
            s_pair += nc;
            lab[r] = best; ub2[r] = b1; lb2[r] = b2 < pruned ? b2 : pruned;
        }
        #pragma omp critical
        for (long l = 0; l < L; l++) stat[2 + l] += dec[l];
        free(P); free(LBv); free(UBv); free(cand); free(dec);
    }
    stat[0] += s_low; stat[1] += s_pair;
}
