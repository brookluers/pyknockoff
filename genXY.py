import numpy as np
import scipy.linalg

rng = np.random.default_rng()

def get_fdpfunc(beta, tol=1e-8):
    abs_beta = np.abs(beta)
    def f(sel):
        return np.sum(abs_beta[sel] < tol) / max(1, np.sum(sel))
    return f

def get_ppvfunc(beta, tol=1e-8):
    abs_beta = np.abs(beta)
    def f(sel):
        return np.sum(abs_beta[sel] > tol) / max(1, np.sum(sel))
    return f

def get_tprfunc(beta, tol=1e-8):
    abs_beta = np.abs(beta)
    k = np.sum(abs_beta > tol)
    def f(sel):
        return np.sum(abs_beta[sel] > tol) / k
    return f

def get_fprfunc(beta, tol=1e-8):
    abs_beta = np.abs(beta)
    p = beta.shape[0]
    k = np.sum(abs_beta > tol)
    def f(sel):
        return np.sum(abs_beta[sel] < tol) / (np.sum(abs_beta[sel] < tol) + p - k)
    return f

def get_fwefunc(beta, tol=1e-8):
    abs_beta = np.abs(beta)
    p = beta.shape[0]
    def f(sel):
        return np.sum(np.sum(abs_beta[sel] < tol) > 0)
    return f

def get_exch(p, rho):
    ret = np.eye(p)
    ret[np.tril_indices(p, -1)] = rho
    ret[np.triu_indices(p, 1)] = rho
    return ret

def get_ar(p, rho):
    gd = np.indices((p,p))
    return rho**np.abs(gd[0]-gd[1])

def get_2block(p, k, rho):
    ret = np.eye(p)
    ret[0:k, k:] = rho * np.eye(k)
    ret[k:, 0:k] = rho * np.eye(k)
    return ret

def gen_Y(X, N, beta, sigma=1):
    # np.random.normal(size=N, scale=sigma)
    return np.dot(X, beta) + rng.normal(scale=sigma, size=N)


def gen_X(N, p, SigmaChol, scale=True, center=True):
    X = rng.standard_normal(size=(N,p))
    # np.random.normal(size = (N, p))
    X = np.matmul(X, SigmaChol.T)
    if center:
        xmeans = np.mean(X, axis=0)
        X = X - xmeans
    if scale:
        xnorms = scipy.linalg.norm(X, axis=0)
        X = X / xnorms
    return X

def gen_X_given_Gram(N, p, SigmaChol, scale=True, center=True):
    X = rng.standard_normal(size=(N,p))
    #np.random.normal(size = (N, p))
    X, _, _ = np.linalg.svd(X, 0)
    X = np.dot(X, SigmaChol.T)
    if center:
        xmeans = np.mean(X, axis=0)
        X = X - xmeans
    if scale:
        xnorms = scipy.linalg.norm(X, axis=0)
        X = X / xnorms
    return X

def rand_beta_flat(p, k, effsize):
    #np.random.choice(np.arange(p), size=k, replace=False)
    nonz_ix = rng.choice(np.arange(p), size=k, replace=False)
    beta = np.zeros(p)
    beta[nonz_ix] = effsize
    # np.random.binomial(1, 0.5, size=k) * 2 - 1
    signs = rng.binomial(n=1,p=0.5,size=k) * 2 - 1
    beta[nonz_ix] *= signs
    return beta

def rand_beta_seq(p, k, bmin, bmax):
    nonz_ix = rng.choice(np.arange(p), size=k, replace=False)
    beta = np.zeros(p)
    beta[nonz_ix] = np.linspace(bmin, bmax, k)
    signs = rng.binomial(n=1, p=0.5, size=k) * 2 - 1
    beta[nonz_ix] *= signs
    return beta

def fix_beta_first_k(p, k, effsize):
    beta = np.concatenate((np.ones(k), np.zeros(p - k)))
    beta *= effsize
    return beta
