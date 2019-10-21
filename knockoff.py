import numpy as np
from scipy.optimize import minimize, Bounds, check_grad, approx_fprime
import scipy.linalg
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLarsIC

def get_alphas_lasso(cp2p, n_alphas, N):
    alpha_max = np.max(np.abs(cp2p)) / N
    alpha_min = alpha_max / 1000
    k = np.linspace(0, n_alphas - 1, n_alphas) / n_alphas
    return alpha_max * (alpha_min / alpha_max)**k

def get_ldetfun(Sigma, tol = 1e-8):
    i, j = np.diag_indices(Sigma.shape[0])
    def f(svec):
        if np.any(svec < tol):
            return -np.Inf
        W = 2 * Sigma
        W[i,j] -= svec
        sgn, ld = np.linalg.slogdet(W)
        if sgn <= 0:
            return -np.Inf
        else:
            return np.sum(np.log(svec)) + ld
    return f

def get_ldetgrad(Sigma):
    p = Sigma.shape[0]
    i, j = np.diag_indices(p)
    def f(svec):
        if np.any(np.isnan(svec)):
            return np.repeat(np.nan, p)
        W = 2 * Sigma
        W[i,j] -= svec
        Winv = np.linalg.inv(W)
        return 1.0 / svec - np.diag(Winv)
    return f


def get_svec_equi(G, minEV = None):
    if minEV is None:
        minEV = scipy.linalg.eigvalsh(G, eigvals=(0,0))[0]
    pdim = G.shape[1]
    svec = np.repeat(min([1.0, 2 * minEV]), pdim)
    return svec


def get_svec_ldet(G, tol=1e-8, maxiter=2000, minEV = None, startval=None, verbose=False, eta=0.3):
    ldetf = get_ldetfun(G)
    ldetgrad = get_ldetgrad(G)
    pdim = G.shape[0]
    if startval is None:
        if minEV is None:
            minEV = scipy.linalg.eigvalsh(G, eigvals=(0,0))[0]
        startval = np.repeat(minEV, pdim)
    ldopt = minimize(lambda x: -ldetf(x),
                     x0 = startval,
                     method='TNC',
                     jac = lambda x: -ldetgrad(x),
                     options={"maxiter": maxiter,
                            'ftol': tol,
                            'eta': eta,
                            'disp': verbose},
                     bounds = Bounds(0.0, 1.0))
    svec = ldopt.x
    return svec


def get_util_random(Qx, N, p, *args):
    Utilde_raw = np.random.rand(N, p)
    Utilde_raw -= np.matmul(Qx, np.matmul(Qx.T, Utilde_raw))
    Utilde, Ru = np.linalg.qr(Utilde_raw, mode='reduced')
    return Utilde

def norm2_utheta_y(theta, ut1, ut2, Y, ut1T_Y=None, ut2T_Y=None):
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    if ut1T_Y is None:
        ut1T_Y = np.matmul(ut1.T, Y)
    if ut2T_Y is None:
        ut2T_Y = np.matmul(ut2.T, Y)
    ut1T_Ysq = np.inner(ut1T_Y, ut1T_Y)
    ut2T_Ysq = np.inner(ut2T_Y, ut2T_Y)
    u12Ycross = np.inner(ut2T_Y, ut1T_Y)
    return sintheta**2 * ut1T_Ysq + costheta**2 * ut2T_Ysq + 2 * costheta * sintheta * u12Ycross


def get_utheta_fixfrac(Qx, N, p, Y, Rx, tseq=None, target_frac=None, ut1=None):
    if tseq is None:
        tseq = np.linspace((1/4)*np.pi, (3/4)*np.pi, 100)
    y2n = np.sum(Y**2)
    QtY = np.matmul(Qx.T, Y)
    Yresid = Y - np.matmul(Qx, QtY)
    ssq_Yresid = np.sum(Yresid**2)
    if target_frac is None:
        G = np.matmul(Rx.T, Rx)
        bhat = scipy.linalg.solve_triangular(Rx, QtY)
        sig2hat = ssq_Yresid / (N - p)
        target_frac = (sig2hat * p) / ((N-p) * sig2hat + np.matmul(np.matmul(bhat, G), bhat))
    Yresid /= np.sqrt(ssq_Yresid)
    if ut1 is None:
        ut1 = get_util_random(Qx, N, p)
    Q_xu = np.concatenate((Qx, ut1), axis=1)
    R_xu = np.eye(Q_xu.shape[1])
    Q_xuy, _ = scipy.linalg.qr_insert(Q_xu, R_xu, Yresid[:, None], Q_xu.shape[1], which='col')
    # ut2 is orthogonal to Y, X, ut1
    Z_Np = np.random.rand(N, p)
    ut2 = np.empty_like(Z_Np)
    np.copyto(ut2, Z_Np)
    ut2 -= np.matmul(Q_xuy, np.matmul(Q_xuy.T, ut2))
    ut2, _ = np.linalg.qr(ut2, mode='reduced')
    # ut3 is orthogonal to ut1, x
    # but it contains Yresid
    ut3 = np.concatenate((Yresid[:, None], Z_Np[:, 0:(p-1)]), axis=1)
    ut3 -= np.matmul(Q_xu, np.matmul(Q_xu.T, ut3))
    ut3, _ = np.linalg.qr(ut3, mode='reduced')
    ut1T_Y = np.matmul(ut1.T, Y)
    frac2 = norm2_utheta_y(tseq, ut1, ut2, Y, ut1T_Y) / y2n
    frac3 = norm2_utheta_y(tseq, ut1, ut3, Y, ut1T_Y) / y2n
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


def stat_lasso_coef(X, Xk, Y, precompute='auto', n_alphas = 100, nfold=3, copy_X=True):
    p = X.shape[1]
    N = X.shape[0]
    XXk = np.concatenate((X, Xk), axis=1)
    lfit = LassoCV(cv=nfold,
            #alphas=alphas,
            n_alphas = n_alphas,
            selection='random', tol=1e-3,
            n_jobs = 2,
            precompute=precompute,
            copy_X = copy_X,
            fit_intercept=False).fit(XXk, Y)
    b = lfit.coef_
    return np.array([abs(b[i]) - abs(b[i + p]) for i in range(p)])


def stat_lassoLarsIC_coef(X, Xk, Y, precompute='auto', copy_X=False, criterion='bic', max_iter=100):
    p = X.shape[1]
    XXk = np.concatenate((X, Xk), axis=1)
    lfit = LassoLarsIC(criterion=criterion,
            precompute=precompute,
            copy_X = copy_X,
            normalize=False,
            max_iter=max_iter,
            eps = 1e-11,
            fit_intercept=False).fit(XXk, Y)
    b = lfit.coef_
    return np.array([abs(b[i]) - abs(b[i + p]) for i in range(p)])



def stat_ols(X, Xk, Y, G2p = None, cp2p = None):
    p = X.shape[1]
    XXk = np.concatenate((X, Xk), axis=1)
    if G2p is None or cp2p is None:
        left = np.matmul(XXk.T,XXk)
        right = np.matmul(XXk.T,Y)
    else:
        left = G2p
        right = cp2p
    try:
        b = scipy.linalg.solve(left, right)
    except (np.linalg.LinAlgError, np.linalg.LinAlgWarning):
        b, _, _, _ = scipy.linalg.lstsq(left, right)
    finally:
        b, _, _, _ = scipy.linalg.lstsq(left, right)
    return np.array([abs(b[i]) - abs(b[i + p]) for i in range(p)])


def stat_crossprod(X, Xk, Y, cp2p=None):
    p = X.shape[1]
    if cp2p is None:
        aXYcp = np.abs(np.matmul(X.T, Y))
        aXkYcp = np.abs(np.matmul(Xk.T, Y))
    else:
        aXYcp = np.abs(cp2p[0:p])
        aXkYcp = np.abs(cp2p[p:(2*p)])
    return np.array(aXYcp - aXkYcp)


def knockoff_threshold(Wstat, q, offset):
    Wabs = np.sort(np.abs(Wstat))
    Wa_mat, Wjmat = np.meshgrid(Wabs, Wstat, indexing='ij')
    numer = offset + np.sum(Wjmat <= -Wa_mat, axis=1)
    denom = np.maximum(1, np.sum(Wjmat >= Wa_mat, axis=1))
    ok_pos = (numer / denom) <=  q
    if np.sum(ok_pos) == 0:
        return float('Inf')
    else:
        return Wabs[np.argmax(ok_pos)]

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
    G = np.matmul(X.T, X)
    Qx, _ = np.linalg.qr(X, mode='reduced')
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
    elif wstat == 'lasso_coefIC':
        W = stat_lassoLarsIC_coef(X, Xtilde, Y)
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
    w[w < tol] = 0
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
