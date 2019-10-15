import numpy as np
import knockoff as ko
import genXY as gen
import scipy.linalg
from scipy.optimize import minimize, Bounds, check_grad, approx_fprime
import timeit

def power_method(A, p, niter=10):
    ek = np.random.normal(size=(p,))
    ek = ek / np.linalg.norm(ek)
    for _ in range(niter):
        ek1 = np.dot(A, ek)
        ek1_norm = np.linalg.norm(ek1)
        ek = ek1 / ek1_norm
    return np.dot(np.dot(ek.T, A), ek)

if __name__ == "__main__":
    p = 200
    rho = 0.9
    print("p={:d}".format(p))
    print("correlation = {:.2f}".format(rho))
    Sigma = gen.get_exch(p, rho)
    n = 5000
    print("n={:d}".format(n))
    SigmaChol = np.linalg.cholesky(Sigma)
    np.random.seed(1)
    X = gen.gen_X(n, p, SigmaChol) # scaled and centered
    G = np.dot(X.T,X)
    meig_exact = scipy.linalg.eigvalsh(G, eigvals=(0,0))[0]
    print("meig_exact = " + str(meig_exact))
    Ginv = scipy.linalg.inv(G)
    power_rslt = []
    test_iter = np.arange(50) + 1
    print("power iterations: ")
    for niter in test_iter:
        cv = 1 / power_method(Ginv, p, niter)
        power_rslt.append(cv)
        print("niter = " + str(niter) + ", ev = " + str(cv))
    print('relative errors: ')
    print(np.abs(power_rslt - meig_exact) / meig_exact)

    st1 = '''
Ginv = scipy.linalg.inv(G)
    '''

    st2 = '''
Ginv = scipy.linalg.inv(G)
meig_iter = 1 / power_method(Ginv, p, niter=30)
    '''

    st3 = '''
Ginv = scipy.linalg.inv(G)
alleigs = np.linalg.eigvalsh(G)
    '''

    st4 = '''
Ginv = scipy.linalg.inv(G)
firsteig = scipy.linalg.eigvalsh(G, eigvals=(0,0))[0]
    '''

    nrep = 100
    tlist = [timeit.timeit(stmt=stj, number=nrep, globals=globals()) for stj in [st1, st2, st3, st4]]
    tnames = ['Ginv only', 'Ginv and 30 power iterations',
              'Ginv and all eigs','Ginv and first eig via eigvalsh']
    for ttl, tj in zip(tnames, tlist):
        print(ttl + "\n\t" + str(tj) + ", average of " + str(tj / nrep))
