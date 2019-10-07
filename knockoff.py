import numpy as np
from scipy.optimize import minimize
import scipy.linalg


def get_ldetfun(Sigma, tol=1e-16):
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
    #print("Maximizing log-determinant of augmented Gram matrix")
    #print("\tInitial steps without using gradient...")
    init_opt = minimize(lambda x: -ldetf(x),
                x0=np.random.uniform(0.0,0.003,size=pdim),
                constraints=scipy.optimize.LinearConstraint(np.identity(pdim),lb=0,ub=1.0),
                options = {'maxiter': 10})
    #print("\tGradient-based maximization with starting value\n\t")
    #print(init_opt.x)
    ldopt = minimize(lambda x: -ldetf(x),
            x0 = init_opt.x,
            jac = lambda x: -ldetgrad(x),
            options={"maxiter": 25000},
            tol = 1e-10,
            constraints = scipy.optimize.LinearConstraint(np.identity(pdim),lb=0,ub=1.0))
    #print("Final steps using gradient...")
    #print("\tresult: \n\t" +
    #        ldopt.message + "\n\tfunciton value: " + str(ldopt.fun) + "\n\tnit = " + str(ldopt.nit) + "\n\tsolution: " + str(ldopt.x))
    svec = ldopt.x
    return svec

def get_util_random(Qx, N, p):
    Utilde_raw = np.random.normal(size = (N, p))
    Utilde_raw = Utilde_raw - np.matmul(Qx, np.dot(Qx.T, Utilde_raw))
    Utilde, _ = scipy.linalg.qr(Utilde_raw, mode='economic')
    return Utilde

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
    Wabs = np.sort([a for a in map(abs, Wstat)])
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
    if stype == 'ldet':
        svec = get_svec_ldet(G)
    elif stype == 'equi':
        svec = get_svec_equi(G)
    else:
        svec = get_svec_ldet(G)
    if wstat == 'ols':
        wfunc = stat_ols
    else:
        wstat = stat_crossprod
    Qx, _ = scipy.linalg.qr(X, mode='economic')
    Xtilde = getknockoffs_qr(X, G, svec, Qx, N, p)
    W = wfunc(X, Xtilde, Y)
    thresh = knockoff_threshold(W, q, offset)
    sel = [W[j] >= thresh for j in range(p)]
    return sel

def get_cmat(X, svec, Ginv=None, tol=1e-7):
    if Ginv is None:
        Ginv = scipy.linalg.solve(G)
    Ginv_S = Ginv * svec
    Smat = np.diag(svec)
    CtC = 2 * Smat - np.matmul(Smat, Ginv_S)
    w, v = scipy.linalg.eigh(CtC)
    w[abs(w) < tol] = 0
    Cmat = np.diag(np.sqrt(w)).dot(v.T)
    return Cmat

def getknockoffs_qr(X, G, svec, Qx,
                    N, p, Utilde=None, Ginv=None, Cmat=None, tol=1e-7):
    if Utilde is None:
        Utilde = get_util_random(Qx, N, p)
    if Ginv is None:
        Ginv = scipy.linalg.solve(G)
    if Cmat is None:
        Cmat = get_cmat(X, svec, Ginv)
    Ginv_S = Ginv * svec
    return X - np.matmul(X, Ginv_S) + np.matmul(Utilde, Cmat)
