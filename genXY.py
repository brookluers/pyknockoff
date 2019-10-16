import numpy as np
import scipy.linalg

def get_exch(p, rho):
    ret = np.eye(p)
    ret[np.tril_indices(p, -1)] = rho
    ret[np.triu_indices(p, 1)] = rho
    return ret

def get_2block(p, k, rho):
    ret = np.eye(p)
    ret[0:k, k:] = rho * np.eye(k)
    ret[k:, 0:k] = rho * np.eye(k)
    return ret

def gen_Y(X, N, beta, sigma=1):
    return np.dot(X, beta) + np.random.normal(size=N, scale=sigma)

def gen_X(N, p, SigmaChol, scale=True, center=True):
    X = np.random.normal(size = (N, p))
    X = np.dot(X, SigmaChol.T)
    if center:
        xmeans = np.mean(X, axis=0)
        X = X - xmeans
    if scale:
        xnorms = scipy.linalg.norm(X, axis=0)
        X = X / xnorms
    return X

def gen_X_given_Gram(N, p, SigmaChol, scale=True, center=True):
    X = np.random.normal(size = (N, p))
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
    nonz_ix = np.random.choice(np.arange(p), size=k, replace=False)
    beta = np.zeros(p)
    beta[nonz_ix] = effsize
    signs = np.random.binomial(1, 0.5, size=k) * 2 - 1
    beta[nonz_ix] *= signs
    return beta

def fix_beta_first_k(p, k, effsize):
    beta = np.concatenate((np.ones(k), np.zeros(p - k)))
    beta *= effsize
    return beta
