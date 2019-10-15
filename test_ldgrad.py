import numpy as np
import knockoff as ko
import genXY as gen
from scipy.optimize import minimize, Bounds, check_grad, approx_fprime

if __name__ == "__main__":
    n = 2000
    p = 80
    rho = 0.2
    print("n={:d}".format(n))
    print("p={:d}".format(p))
    print("correlation = {:.2f}".format(rho))
    Sigma = gen.get_exch(p, rho)
    SigmaChol = np.linalg.cholesky(Sigma)
    np.random.seed(1)
    X = gen.gen_X(n, p, SigmaChol) # scaled and centered
    G = np.dot(X.T,X)

    ldf = ko.get_ldetfun(G)
    ldg = ko.get_ldetgrad(G)
    npts = 100
    sseq = np.linspace(1e-5, 1-1e-5, npts)
    print("evaluating at s[i] * (1,..,1)")
    print("\t for s[i] in :")
    print(sseq)
    appx = []
    mine = []
    ptnorms = np.zeros(npts)
    for i in np.arange(npts):
        svi = np.repeat(sseq[i], p)
        ptnorms[i] = np.sqrt(np.sum(svi**2))
        appx.append(approx_fprime(svi, ldf, 1e-8))
        mine.append(ldg(svi))
    err_rel = np.array([np.sqrt(np.sum((a-b)**2)) for a, b in zip(mine, appx)]) / [np.sqrt(np.sum(a**2)) for a in mine]
    err_rel2 = np.array([np.sqrt(np.sum((a-b)**2)) for a, b in zip(mine, appx)]) / [np.sqrt(np.sum(a**2)) for a in appx]
    gcor = np.corrcoef(np.array(mine),np.array(appx))[p:, 0:p]
    abs_err = [np.sqrt(np.sum((a-b)**2)) for a, b in zip(mine, appx)]
    print("\nerrors |appx - my_func| ")
    print(abs_err)
    print("Relative error, |appx - my_func| / |my_func|")
    print(err_rel)
    print("Relative error, |appx - my_func| / | appx |")
    print(err_rel2)
    print("\ncomponent-wise correlations between (appx, my_func)")
    print(gcor[np.diag_indices(p)])
    worst_ix = np.argmax(abs_err)
    print("Worst absolute error: " + str(abs_err[worst_ix]))
    print("\t s[i] = " + str(sseq[worst_ix]))
    print("\t appx = " + str(appx[worst_ix]))
    print("\t mine = " + str(mine[worst_ix]))
