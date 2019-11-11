import numpy as np
import scipy.linalg
import scipy.stats


rng = np.random.default_rng()

def bhPvals(x, q):
    x = np.array(x)
    m = np.size(x,0)
    xisort = np.argsort(x)
    z = np.zeros(m)
    rej = np.repeat(False, m)
    rej_ix = xisort[x[xisort] <= (np.linspace(1,m,m) / m) * q]
    rej[rej_ix] = True
    return rej

def bhOLSReg(X, Y, q):
    n, p = X.shape
    Qx, Rx = np.linalg.qr(X, mode='reduced')
    QtY = np.matmul(Qx.T, Y)
    resids = Y - np.matmul(Qx, QtY)
    sig2hat = np.sum(resids**2) / (n - p)
    Rinv = scipy.linalg.solve_triangular(Rx, np.eye(p))
    bhat = np.matmul(Rinv, QtY)
    bse = np.sqrt(sig2hat) * np.linalg.norm(Rinv, axis=1)
    ts = bhat / bse
    pvals = 2 * (1.0 - scipy.stats.t.cdf(np.abs(ts), df=n-p))
    return bhPvals(pvals, q)
