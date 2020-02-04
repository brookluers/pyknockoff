import numpy as np
from scipy.optimize import minimize, Bounds, check_grad, approx_fprime
import scipy.linalg
import cvxpy as cp
import warnings
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoLarsIC

rng = np.random.default_rng()

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

def get_svec_sdp(G, minEV = None):
    p = G.shape[0]
    s = cp.Variable(p)
    objective = cp.Minimize(cp.sum(1 - s))
    sdcon = (2 * G - cp.diag(s) >> 0)
    constraints = [0 <= s, s <= 1, sdcon]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CVXOPT)
    return s.value

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

def get_util_fixed(Qx, N, p, *args):
    znp = np.zeros((N, p))
    utraw = np.concatenate((Qx, znp), axis=1)
    Qx0, Rx0 = np.linalg.qr(utraw, mode='reduced')
    return Qx0[:,p:]

def get_util_random(Qx, N, p, *args):
    Utilde_raw = rng.standard_normal(size=(N,p))
    Utilde_raw -= np.matmul(Qx, np.matmul(Qx.T, Utilde_raw))
    Utilde, Ru = np.linalg.qr(Utilde_raw, mode='reduced')
    return Utilde

def get_util_runif(Qx, N, p, *args):
    Z = rng.standard_normal(size=(N, p))
    d, v = np.linalg.eig(np.matmul(Z.T, Z))
    # (Z^t Z)^(-1/2)
    zti = np.matmul(v * 1 / np.sqrt(d), v.T)
    U0 = np.matmul(Z, zti)
    U0 -= np.matmul(Qx, np.matmul(Qx.T, U0))
    Util, _ = np.linalg.qr(U0, mode='reduced')
    return Util


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
        tseq = np.linspace((1/4)*np.pi, (3/4)*np.pi, 200)
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
    Z_Np = rng.standard_normal(size=(N,p))
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

def stat_paired_diff(b2p, p):
    return np.array([abs(b2p[i]) - abs(b2p[i + p]) for i in range(p)])

def lasso_coef(X, Xk, Y, n_alphas = 100, nfold = 3, copy_X=True):
    p = X.shape[1]
    N = X.shape[0]
    XXk = np.concatenate((X, Xk), axis=1)
    lfit = LassoCV(cv=nfold,
            n_alphas = n_alphas,
            tol=1e-3,
            n_jobs = 2,
            copy_X = copy_X,
            normalize = False,
            fit_intercept=False).fit(XXk, Y)
    b = lfit.coef_
    return b

def stat_lasso_coef(X, Xk, Y, n_alphas = 100, nfold=3, copy_X = True):
    p = X.shape[1]
    b2p = lasso_coef(X, Xk, Y, n_alphas, nfold, copy_X)
    return stat_paired_diff(b2p, p)

def ridge_coef(X, Xk, Y, n_alphas=20):
    N, p = X.shape
    XXk = np.concatenate((X, Xk), axis=1)
    cp2p = np.matmul(XXk.T,Y)
    alphas = get_alphas_lasso(cp2p, n_alphas, N)
    rfit = RidgeCV(alphas, fit_intercept=False,
                normalize=False).fit(XXk, Y)
    b = rfit.coef_
    return b

def stat_ridge_coef(X, Xk, Y, n_alphas = 20):
    p = X.shape[1]
    b = ridge_coef(X, Xk, Y, n_alphas)
    return stat_paired_diff(b, p)

def lassoLarsIC_coef(X, Xk, Y, precompute='auto', copy_X=True, criterion='aic'):
    p = X.shape[1]
    XXk = np.concatenate((X, Xk), axis=1)
    lfit = LassoLarsIC(criterion=criterion,
            copy_X = copy_X,
            normalize = False,
            eps = 1e-11,
            fit_intercept=False).fit(XXk, Y)
    b = lfit.coef_
    return b

def stat_lassoLarsIC_coef(X, Xk, Y, precompute='auto', copy_X=True, criterion='aic'):
    p = X.shape[1]
    b2p = lassoLarsIC_coef(X, Xk, Y, precompute, copy_X, criterion)
    return stat_paired_diff(b2p, p)

def ols_coef(X, Xk, Y, G2p=None, cp2p=None):
    p = X.shape[1]
    XXk = np.concatenate((X, Xk), axis=1)
    if G2p is None or cp2p is None:
        left = np.matmul(XXk.T,XXk)
        right = np.matmul(XXk.T,Y)
    else:
        left = G2p
        right = cp2p
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            b = scipy.linalg.solve(left, right)
        except: # (scipy.linalg.LinAlgError,    scipy.linalg.LinAlgWarning):
            # b, _, _, _ = scipy.linalg.lstsq(left, right)
            print("singular OLS, returning cross products")
            b = np.concatenate((np.matmul(X.T,Y), np.matmul(Xk.T,Y)))
            #aXYcp = np.abs(np.matmul(X.T, Y))
            #aXkYcp = np.abs(np.matmul(Xk.T, Y))
            #ret = np.array(aXYcp - aXkYcp)
    return b

def stat_ols(X, Xk, Y, G2p = None, cp2p = None):
    p = X.shape[1]
    b2p = ols_coef(X, Xk, Y, G2p, cp2p)
    return stat_paired_diff(b2p, p)

def crossprod_coef(X, Xk, Y, cp2p=None):
    p = X.shape[1]
    if cp2p is None:
        b = np.concatenate((np.matmul(X.T,Y),np.matmul(Xk.T,Y)))
    else:
        b = cp2p
    return b

def stat_crossprod(X, Xk, Y, cp2p=None):
    b2p = crossprod_coef(X, Xk, Y, cp2p)
    p = X.shape[1]
    return stat_paired_diff(b2p, p)

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

def multiKO(nrep, X, Y, G, Ginv, Cmat, svec, Qx,
            N, p, q, wstat, tol=1e-10):
    Tmat = []
    if wstat == 'ols':
        bfunc = ols_coef
    elif wstat == 'ridge':
        bfunc = ridge_coef
    elif wstat=='crossprod':
        bfunc = crossprod_coef
    elif wstat == 'lasso_coef':
        bfunc = lasso_coef
    elif wstat == 'lasso_coefIC':
        bfunc = lassoLarsIC_coef
    else:
        print("unknown wtype, using cross product")
        bfunc = crossprod_coef
    # create many knockoff |beta_j|
    for i in range(nrep):
        xti = getknockoffs_qr(X, G, svec, Qx, N, p,
                    Utilde = None, Ginv=Ginv, Cmat=Cmat,
                    tol=tol)
        b = bfunc(X, xti, Y)
        Tmat.append(np.abs(b[p:]))
    # store the |beta_j| for the original variables
    # from one of the _nrep_ fits
    Tmat.append(np.abs(b[0:p]))
    Tmat = np.array(Tmat) # nrep + 1 by p
    ot = np.argsort(Tmat, axis=0)
    kappa = ot[nrep, :] # indices of the largest magnitudes
    Tsort = np.take_along_axis(Tmat, ot, axis=0)
    tau = Tsort[nrep, :] - Tsort[nrep - 1, :]
    # each row is a threshold
    checktau = [[tau_j >= t for tau_j in tau] for t in tau]
    # true means orig. coef not the largest among
    #   the (k+1) knockoffs + original coefficients
    # i.e. true suggests false discovery
    kbad = [ki <= (nrep - 1) for ki in kappa]
    # true means that the original variable's coefficient
    #   was the largest magnitude coefficient
    ktop = [not kj for kj in kbad]
    numer = [[ktj and ctj for (ktj, ctj) in zip(kbad,  cur_thresh)] for cur_thresh in checktau]
    denom = [[ktj and ctj for (ktj, ctj) in zip(ktop,  cur_thresh)] for cur_thresh in checktau]
    numer = np.array(numer)
    denom = np.array(denom)
    denom_final = np.maximum(np.sum(denom, axis=1),np.repeat(1,p))
    numer_final = (1/nrep) + (1/nrep) * np.sum(numer,axis=1)
    tau[numer_final / denom_final <= q]
    ltq = tau[numer_final / denom_final <= q]
    if len(ltq) > 0:
        thresh = np.min(tau[numer_final / denom_final <= q])
    else:
        thresh = float('Inf')
    sel = np.logical_and(tau >= thresh, ktop)
    return sel

def bootKO(nrep, X, Y, G, Ginv, Cmat, svec, Qx,
            N, p, q, offset, Wfunc, wstat='lasso_coefIC',
            type='bootThresh', tol=1e-10):
    if type == 'multiKO':
        sel_consensus = multiKO(nrep, X, Y, G, Ginv, Cmat, svec, Qx, N, p, q, wstat, tol)
    else:
        selmat = []
        Wmat = []
        for i in range(nrep):
            Xtilde = getknockoffs_qr(X, G, svec, Qx, N, p,
                        Utilde = None, Ginv=Ginv, Cmat=Cmat,
                        tol=tol)
            W = Wfunc(X, Xtilde, Y)
            Wmat.append(W)
            thresh = knockoff_threshold(W, q, offset)
            sel_i = np.array([W[j] >= thresh for j in range(p)])
            selmat.append(sel_i)
        selmat = np.array(selmat)
        Wmat = np.array(Wmat)
        sel_consensus = np.repeat(False, p)
        if type == 'avg_nsel':
            ranksel = np.argsort(np.sum(selmat, axis=0))
            avg_nsel = int(np.round(np.mean(np.sum(selmat, axis=1))))
            if avg_nsel >= 1:
                sel_consensus[ranksel[-avg_nsel:]] = True
        else:
            ngrid = 20
            wmax = np.max(Wmat)
            testThresh = np.linspace(1e-7, wmax, ngrid)
            tmax = 1e-6
            prevMax = wmax # prevMax will decrease
            while (abs(prevMax - tmax) / prevMax) > 1e-4:
                bootNumer = np.array([np.mean(np.sum(Wmat <= -tj, axis=1)) for tj in testThresh])
                bootDenom = np.array([np.mean(np.sum(Wmat >= tj, axis=1)) for tj in testThresh])
                anyLess = (bootNumer / bootDenom) <= q
                if np.sum(anyLess) < 1:
                    tmax = np.inf
                    break
                tix = np.argmax(anyLess)
                prevMax = tmax
                tmax = testThresh[tix]
                testThresh = np.linspace(testThresh[max(0, tix-2)], tmax, ngrid)
            cutoff_ix = int(np.round(np.mean(np.sum(Wmat >= tmax, axis=1))))
            ranksel = np.argsort(np.mean(Wmat >= tmax, axis=0))
            if cutoff_ix > 0:
                sel_consensus[ranksel[-cutoff_ix:]] = True
    return sel_consensus

def doKnockoffSplitSample(X, Y, q, nU = 100, offset=1,
                          stype = 'ldet', svec = None,
                          wstat='ols', scale = True, center=True,
                          tol=1e-10, returnW = False):
    N, p = X.shape
    ixN = np.arange(N)
    ixnew = rng.permutation(ixN)
    if N % 2 != 0:
        ixnew = ixnew[0:-1]
        X = X[ixnew,:]
        N = X.shape[0]
    X1 = X[ixnew[0:int(N/2)],:]
    X2 = X[ixnew[int(N/2):],:]
    if center:
        x1means = np.mean(X1, axis=0)
        x2means = np.mean(X2, axis=0)
        X1 = X1 - x1means
        X2 = X2 - x2means
    if scale:
        x1norms = scipy.linalg.norm(X1, axis=0)
        x2norms = scipy.linalg.norm(X2, axis=0)
        X1 = X1 / x1norms
        X2 = X2 / x2norms
    Y1 = Y[ixnew[0:int(N/2)]]
    Y2 = Y[ixnew[int(N/2):]]
    X12 = np.concatenate((X1,X2), axis=1)
    Q12 , _ = np.linalg.qr(X12, mode='reduced')
    Q1, R1 = np.linalg.qr(X1, mode='reduced')
    Q2, R2 = np.linalg.qr(X2, mode='reduced')
    G1 = np.matmul(R1.T, R1)
    G2 = np.matmul(R2.T, R2)
    G1i = scipy.linalg.inv(G1)
    G2i = scipy.linalg.inv(G2)
    if stype == 'ldet':
        sv1 = get_svec_ldet(G1)
        sv2 = get_svec_ldet(G2)
    elif stype == 'equi':
        sv1 = get_svec_equi(G1)
        sv2 = get_svec_equi(G2)
    elif stype == 'sdp':
        sv1 = get_svec_sdp(G1)
        sv2 = get_svec_sdp(G2)
    else:
        sv1 = get_svec_ldet(G1)
        sv2 = get_svec_ldet(G2)
    C1 = get_cmat(X1, sv1, G1, G1i, tol)
    C2 = get_cmat(X2, sv2, G2, G2i, tol)
    if wstat == 'ols':
        Wfunc = stat_ols
    elif wstat == 'ridge':
        Wfunc = stat_ridge_coef
    elif wstat=='crossprod':
        Wfunc = stat_crossprod
    elif wstat == 'lasso_coef':
        Wfunc = stat_lasso_coef
    elif wstat == 'lasso_coefIC':
        Wfunc = stat_lassoLarsIC_coef
    else:
        print("unknown wtype, using cross product")
        Wfunc = stat_crossprod
    Ut1best = None
    nselbest = -1
    selbest = None
    for i in np.arange(nU):
        Utilde_raw = rng.standard_normal(size=X1.shape)
        Utilde_raw -= np.matmul(Q12, np.matmul(Q12.T, Utilde_raw))
        Utilde, Ru = np.linalg.qr(Utilde_raw, mode='reduced')
        Xtilde1 = getknockoffs_qr(X1, G1, sv1,
                                  Q1, X1.shape[0], p, Utilde, G1i, C1)
        W1 = Wfunc(X1, Xtilde1, Y1)
        thresh = knockoff_threshold(W1, q, offset)
        sel = np.array([W1[j] >= thresh for j in range(p)])
        nsel_cur = np.sum(sel)
        if nsel_cur > nselbest:
            nselbest = nsel_cur
            selbest = sel
            Ut1best = Utilde
    Xtilde2 = getknockoffs_qr(X2, G2, sv2,
                              Q2, X2.shape[0], p, Ut1best, G2i, C2)
    W2 = Wfunc(X2, Xtilde2, Y2)
    thresh2 = knockoff_threshold(W2, q, offset)
    sel2 = np.array([W2[j] >= thresh2 for j in range(p)])
    if returnW:
        return sel2, W2
    else:
        return sel2

def doKnockoff(X, Y, q, offset=1,
                stype='ldet', svec=None, wstat='ols',
                scale = True, center=True,
                utype = 'random',
                Utilde = None, nrep=1,
                Qx=None, Rx=None, Ginv=None, G=None,
                Cmat=None,
                tol=1e-10, returnW = False,
                bootType = 'bootThresh',
                rUUYf = False,
                returnS = False):
    N, p = X.shape
    if center:
        xmeans = np.mean(X, axis=0)
        X = X - xmeans
    if scale:
        xnorms = scipy.linalg.norm(X, axis=0)
        X = X / xnorms
    if Qx is None:
        Qx, Rx = np.linalg.qr(X, mode='reduced')
    if G is None:
        G = np.matmul(Rx.T, Rx)
    if svec is None:
        if stype == 'ldet':
            svec = get_svec_ldet(G)
        elif stype == 'equi':
            svec = get_svec_equi(G)
        elif stype == 'sdp':
            svec = get_svec_sdp(G)
        else:
            svec = get_svec_ldet(G)
    if Cmat is None:
        Cmat = get_cmat(X, svec, G, Ginv, tol)
    if wstat == 'ols':
        Wfunc = stat_ols
    elif wstat == 'ridge':
        Wfunc = stat_ridge_coef
    elif wstat=='crossprod':
        Wfunc = stat_crossprod
    elif wstat == 'lasso_coef':
        Wfunc = stat_lasso_coef
    elif wstat == 'lasso_coefIC':
        Wfunc = stat_lassoLarsIC_coef
    else:
        print("unknown wstat, using cross product")
        wstat = 'crossprod'
        Wfunc = stat_crossprod
    if nrep < 2:
        if Utilde is None:
            if utype == 'random':
                Utilde = get_util_random(Qx, N, p)
            elif utype == 'runif':
                Utilde = get_util_runif(Qx, N, p)
            elif utype == 'fixed':
                Utilde = get_util_fixed(Qx, N, p)
            elif utype == 'varfrac':
                Utilde = get_utheta_fixfrac(Qx, N, p,Y,Rx)
            else:
                print("utype not recognized, using random")
                Utilde = get_util_random(Qx, N, p)
        Xtilde = getknockoffs_qr(X, G, svec, Qx, N, p,
                    Utilde, Ginv, Cmat)
        W = Wfunc(X, Xtilde, Y)
        thresh = knockoff_threshold(W, q, offset)
        sel = np.array([W[j] >= thresh for j in range(p)])
    else:
        if utype == 'split':
            sel, W = doKnockoffSplitSample(X, Y, q, nU=nrep,
                                           offset=offset, stype=stype,
                                           svec = svec, wstat=wstat,
                                           scale=True, center=True,
                                           returnW = True)
        else:
            sel = bootKO(nrep, X, Y, G, Ginv, Cmat, svec,
                    Qx, N, p, q,
                    offset, Wfunc, wstat,
                    type=bootType, tol=tol)
    if returnW and (nrep < 2 or utype=='split'):
        return sel, W
    elif rUUYf and nrep < 2:
        uuy2 = np.sum(np.matmul(Utilde.T,Y)**2)
        yn2 = np.sum(Y**2)
        uuyf = uuy2 / yn2
        return sel, np.array([uuyf])
    elif returnS:
        return sel, svec
    else:
        return sel


def get_cmat(X, svec, G=None, Ginv=None, tol=1e-7):
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
