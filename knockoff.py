import numpy as np
from scipy.optimize import minimize, Bounds
import scipy.linalg
from sklearn.linear_model import LassoCV


def get_ldetfun(Sigma, tol=1e-10):
    def f(svec):
        W = 2 * Sigma - np.diag(svec)
        Wev = np.linalg.eigvalsh(W)
        if any(Wev < tol):
            return -np.Inf
        else:
            return np.sum(np.log(svec)) + np.sum(np.log(Wev))
    return f

def get_ldetgrad(Sigma):
    pdim = Sigma.shape[1]
    def f(svec):
        W = 2 * Sigma - np.diag(svec)
        Winv = np.linalg.inv(W)
        return 1.0 / svec - np.diag(Winv)
    return f

def get_svec_equi(G):
    evs = np.linalg.eigvalsh(G)
    pdim = G.shape[1]
    svec = np.repeat(min([1.0, 2 * evs.min()]), pdim)
    return svec

def get_svec_ldet(G):
    ldetf = get_ldetfun(G)
    ldetgrad = get_ldetgrad(G)
    pdim = G.shape[0]
    init_opt = minimize(lambda x: -ldetf(x),
                x0=np.random.uniform(0.0,0.003,size=pdim),
                method='L-BFGS-B',
                bounds = Bounds(0.0, 1.0),
                options = {'maxiter': 10})
    ldopt = minimize(lambda x: -ldetf(x),
            x0 = init_opt.x,
            method='L-BFGS-B',
            jac = lambda x: -ldetgrad(x),
            options={"maxiter": 25000},
            tol = 1e-10,
            bounds = Bounds(0.0, 1.0))
    svec = ldopt.x
    return svec

def get_util_random(Qx, N, p, *args):
    Utilde_raw = np.random.normal(size = (N, p))
    Utilde_raw = Utilde_raw - np.matmul(Qx, np.dot(Qx.T, Utilde_raw))
    Utilde, _ = scipy.linalg.qr(Utilde_raw, mode='economic')
    return Utilde

def norm2_utheta_y(theta, ut1, ut2, Y):
    utheta = [np.sin(tj) * ut1 + np.cos(tj) * ut2 for tj in theta]
    return [scipy.linalg.norm(np.dot(utj, np.dot(utj.T,Y)))**2 for utj in utheta]


def get_utheta_fixfrac(Qx, N, p, Y, Rx, tseq=None, target_frac=None):
    if tseq is None:
        tseq = np.linspace((1/4)*np.pi, (3/4)*np.pi, 250)
    y2n = np.sum(Y**2)
    QtY = np.dot(Qx.T, Y)
    Yresid = Y - np.dot(Qx, QtY)
    if target_frac is None:
        G = np.dot(Rx.T, Rx)
        bhat = scipy.linalg.solve_triangular(Rx, QtY)
        sig2hat = np.sum(Yresid**2) / (N - p)
        target_frac = (sig2hat * p) / ((N-p) * sig2hat + np.dot(np.dot(bhat, G), bhat))
    Yresid /= np.sqrt(np.sum(Yresid**2))

    ut1 = get_util_random(Qx, N, p)
    Q_xuy, _ = scipy.linalg.qr(np.concatenate((ut1, Yresid[:, None], Qx), axis=1), mode='economic')
    Q_xu, _ = scipy.linalg.qr(np.concatenate((ut1, Qx), axis=1), mode='economic')

    ut2 = np.random.normal(size = (N, p))
    ut2 -= np.dot(Q_xuy, np.dot(Q_xuy.T, ut2))
    ut2, _ = scipy.linalg.qr(ut2, mode='economic')

    ut3 = np.concatenate((Yresid[:, None], np.random.normal(size = (N, p - 1))), axis=1)
    ut3 -= np.dot(Q_xu, np.dot(Q_xu.T, ut3))
    ut3, _ = scipy.linalg.qr(ut3, mode='economic')

    frac2 = norm2_utheta_y(tseq, ut1, ut2, Y) / y2n
    frac3 = norm2_utheta_y(tseq, ut1, ut3, Y) / y2n
    fd2 = np.abs(frac2 - target_frac)
    fd3 = np.abs(frac3 - target_frac)
    imin2 = fd2.argmin()
    imin3 = fd3.argmin()
    if fd3[imin3] < fd2[imin2]:
        theta = tseq[imin3]
        ut_other = ut3
    else:
        theta = tseq[imin2]
        ut_other = ut2
    return np.sin(theta) * ut1 + np.cos(theta) * ut_other

def stat_lasso_coef(X, Xk, Y, n_alphas = 200):
    p = X.shape[1]
    N = X.shape[0]
    XXk = np.concatenate((X, Xk), axis=1)
    cp = np.dot(XXk.T, Y)
    alpha_max = max(np.abs(cp)) # ? divide by N here ?
    alpha_min = alpha_max / 1000
    k = np.linspace(0, n_alphas - 1, n_alphas) / n_alphas
    alphas = alpha_max * (alpha_min / alpha_max)**k
    lfit = LassoCV(cv=5, alphas=alphas, max_iter = 5000).fit(XXk, Y)
    b = lfit.coef_
    return [abs(b[i]) - abs(b[i + p]) for i in range(p)]

def stat_ols(X, Xk, Y):
    p = X.shape[1]
    XXk = np.concatenate((X, Xk), axis=1)
    b = scipy.linalg.solve(np.dot(XXk.T, XXk), np.dot(XXk.T, Y))
    return [abs(b[i]) - abs(b[i + p]) for i in range(p)]

def stat_crossprod(X, Xk, Y):
    aXYcp = np.abs(np.dot(X.T, Y))
    aXkYcp = np.abs(np.dot(Xk.T, Y))
    return aXYcp - aXkYcp

def knockoff_threshold(Wstat, q, offset):
    p = len(Wstat)
    Wabs = np.sort(np.abs(Wstat))
    ok_ix = []
    for j in range(p):
        thresh = Wabs[j]
        numer = offset + np.sum([Wstat[i] <= -thresh for i in range(p)])
        denom = max(1.0, np.sum([Wstat[i] >= thresh for i in range(p)]))
        if numer / denom <= q:
            ok_ix.append(j)
    if len(ok_ix) > 0:
        return Wabs[ok_ix[0]]
    else:
        return float('Inf')

def doKnockoff(X, Y, q, offset=1,
                stype='ldet', wstat='ols',
                scale = True, center=True, tol=1e-10):
    N, p = X.shape
    if center:
        xmeans = np.mean(X, axis=0)
        X = X - xmeans
    if scale:
        xnorms = scipy.linalg.norm(X, axis=0)
        X = X / xnorms
    G = np.dot(X.T, X)
    Qx, _ = scipy.linalg.qr(X, mode='economic')
    Xtilde = getknockoffs_qr(X, G, svec, Qx, N, p)
    if stype == 'ldet':
        svec = get_svec_ldet(G)
    elif stype == 'equi':
        svec = get_svec_equi(G)
    else:
        svec = get_svec_ldet(G)
    if wstat == 'ols':
        W = stat_ols(X, Xtilde, Y)
    elif wstat=='crossprod':
        W = stat_crossprod(X, Xtilde, Y)
    elif wstat == 'lasso_coef':
        W = stat_lasso_coef(X, Xtilde, Y)
    else:
        W = stat_crossprod(X, Xtilde, Y)
    thresh = knockoff_threshold(W, q, offset)
    sel = [W[j] >= thresh for j in range(p)]
    return sel

def get_cmat(X, svec, Ginv=None, tol=1e-7):
    if Ginv is None:
        Ginv = scipy.linalg.inv(G)
    CtC = Ginv * -np.outer(svec, svec)   # - S Ginv S
    i, j = np.diag_indices(Ginv.shape[0])
    CtC[i, j] += 2 * svec
    # CtC_old = 2 * Smat - np.matmul(Smat, Ginv_S)
    w, v = scipy.linalg.eigh(CtC)
    w[abs(w) < tol] = 0
    Cmat = np.sqrt(w)[:, None] * v.T
    return Cmat

def getknockoffs_qr(X, G, svec, Qx,
                    N, p, Utilde=None, Ginv=None, Cmat=None, tol=1e-7):
    if Utilde is None:
        Utilde = get_util_random(Qx, N, p)
    if Ginv is None:
        Ginv = scipy.linalg.inv(G)
    if Cmat is None:
        Cmat = get_cmat(X, svec, Ginv)
    Ginv_S = Ginv * svec
    return X - np.matmul(X, Ginv_S) + np.matmul(Utilde, Cmat)
