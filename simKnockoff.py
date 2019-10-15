import knockoff as ko
import genXY as gen
import numpy as np
import scipy
import pandas as pd



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

def power_method(A, p, startvec, niter=10):
    ek = startvec
    for _ in range(niter):
        ek1 = np.dot(A, ek)
        ek1_norm = np.linalg.norm(ek1)
        ek = ek1 / ek1_norm
    return np.dot(np.dot(ek.T, A), ek)

def one_rslt(Utilde, W, Y, p, FDR, ppv, tpr, fdp, fpr, offset=1):
    thresh = ko.knockoff_threshold(W, FDR, offset)
    sel = [W[j] >= thresh for j in range(p)]
    uf1 = np.dot(Utilde, np.dot(Utilde.T, Y))
    ufrac = np.sum(uf1**2) / np.sum(Y**2)
    ret = {
    'ufrac': ufrac,
    'ppv': ppv(sel),
    'tpr': tpr(sel),
    'fdp': fdp(sel),
    'fpr': fpr(sel),
    'nsel': np.sum(sel)
    }
    ret.update({"sel{:d}".format(i): 1 * sel[i] for i in range(p)})
    return ret

def kosim(nsim_x, nsim_yx, nsim_uyx, N, p, k, rho,
            effsize, FDR, offset=1, corstr='exch', betatype='flat',
            stypes = ['equi', 'ldet'], wtypes = ['ols', 'crossprod'],
            utypes = ['utheta', 'util_rand'],
            scale=True, center=True,
            target_ufrac = None,
             fixGram = False, to_csv = True):
    print("\n---------\n")
    sfunc_d = {}
    for stype in stypes:
        if stype == 'equi':
            sfunc_d[stype] = lambda G, minEV: ko.get_svec_equi(G, minEV=minEV)
        elif stype == 'ldet':
            sfunc_d[stype] = lambda G, minEV: ko.get_svec_ldet(G, minEV = minEV)
    nstypes = len(sfunc_d)
    snames = list(sfunc_d.keys())
    utfunc_d = {}
    for utype in utypes:
        if utype == 'util_rand':
            utfunc_d[utype] = ko.get_util_random
        elif utype == 'utheta':
            tseq = np.linspace((1/4)*np.pi, (3/4)*np.pi, 250)
            if target_ufrac is None:
                print("Target fraction of UUY variance:")
                print("\that(sigma^2)*p / ((N-p)*hat(sigma^2) + bhat^t G bhat)")
            else:
                print("Target fraction of UUY variance: {:.3f}".format(target_ufrac))
            utfunc_d[utype] = lambda Qx, N, p, Y, Rx: ko.get_utheta_fixfrac(Qx, N, p, Y, Rx, tseq, target_ufrac)
    nutypes = len(utfunc_d)
    utnames = list(utfunc_d.keys())
    wfunc_d = {}
    for wtype in  wtypes:
        if wtype == 'ols':
            wfunc_d[wtype] = ko.stat_ols
        elif wtype == 'crossprod':
            wfunc_d[wtype] = ko.stat_crossprod
        elif wtype == 'lasso_coef':
            wfunc_d[wtype] = ko.stat_lasso_coef
    wnames = list(wfunc_d.keys())
    if betatype == 'flat':
        beta = gen.rand_beta_flat(p, k, effsize)
    elif betatype == 'firsthalf':
        print("First {:d} coefficients nonzero".format(p//2))
        beta = gen.fix_beta_firsthalf(p, effsize)
    print("true effects = ")
    print(beta)
    ppv = get_ppvfunc(beta)
    tpr = get_tprfunc(beta)
    fdp = get_fdpfunc(beta)
    fpr = get_fprfunc(beta)
    if fixGram:
        print("Generating X with fixed Gram matrix")
        genXfunc = gen.gen_X_given_Gram
    else:
        genXfunc = gen.gen_X
    if corstr == 'exch':
        print("Exchangeable covariance matrix")
        Sigma = gen.get_exch(p, rho)
    elif corstr == '2block':
        print("Using '2block' covariance matrix")
        Sigma = gen.get_2block(p, rho)
    else:
        Sigma = gen.get_exch(p, rho)
    print("cov(X)[0:5, 0:5]: ")
    np.set_printoptions(precision=3, suppress=True)
    print(Sigma[0:5, 0:5])
    SigmaChol = np.linalg.cholesky(Sigma)
    genYfunc = lambda X, N: gen.gen_Y(X, N, beta)

    W_s_names = [(wnames[r1], snames[r2])
             for r1 in range(len(wnames)) for r2 in range(nstypes)]
    if offset == 1:
        print("Knockoff+, offset = {:d}".format(offset))
    elif offset == 0:
        print("Knockoff, offset = {:d}".format(offset))
    else:
        print("offset must be 0 or 1, setting to 1")
        offset = 1

    # Use power iterations to approximate
    # the smallest eigenvalue of X^t X
    # use for log-det and equivariant
    # tuning of S
    rand_unit = np.random.normal(size=(p,))
    rand_unit = rand_unit / np.linalg.norm(rand_unit)

    rslt = []
    for jx in range(nsim_x):
        X = genXfunc(N, p, SigmaChol, scale, center)
        Qx, Rx = scipy.linalg.qr(X, mode='economic')
        G = np.dot(Rx.T, Rx) # = X^t X
        Ginv = scipy.linalg.inv(G)
        minEV = 1 / power_method(Ginv, p,
                                    startvec = rand_unit, niter = 30)
        slist = [sfunc_d[stype](G, minEV) for stype in sfunc_d]
        cmlist = [ko.get_cmat(X, sv, Ginv) for sv in slist]
        for jyx in range(nsim_yx):
            Y = genYfunc(X, N)
            for juyx in range(nsim_uyx):
                Utlist = [utfunc_d[utype](Qx, N, p, Y, Rx) for utype in utfunc_d]
                for ut_ix in range(nutypes):
                    Ut = Utlist[ut_ix]
                    Xtlist = [ko.getknockoffs_qr(X, G, slist[r], Qx, N, p, Ut, Ginv, cmlist[r]) for r in range(nstypes)]
                    Wlist = [wfunc_d[wtype](X, Xk, Y)
                             for wtype in wfunc_d for
                             Xk in Xtlist]
                    res_byW = [one_rslt(Ut, wvec, Y, p, FDR,
                                    ppv, tpr, fdp, fpr, offset) for wvec in Wlist]
                    for wr_ix in range(len(res_byW)):
                        res_byW[wr_ix].update({
                        'juyx': juyx, 'jyx': jyx, 'jx': jx,
                        'N': N, 'p': p, 'k': k, 'FDR': FDR,
                        'rho': rho, 'corstr': corstr,
                        'utype': utnames[ut_ix],
                        'wstat': W_s_names[wr_ix][0],
                        'stype': W_s_names[wr_ix][1],
                        'offset': offset
                        })
                    rslt.append(res_byW)

    df = pd.concat([pd.DataFrame(rj) for rj in rslt])
    if to_csv:
        df.to_csv("ko-x{:d}-yx{:d}-uyx{:d}-".format(nsim_x,nsim_yx,nsim_uyx) + corstr + "-" + betatype + "-N{:d}-p{:d}-rho{:.2f}-off{:d}-fixGram".format(N, p, rho, offset) + str(fixGram) + ".csv", index=False)
    return df


if __name__ == "__main__":

    # Sample size
    n = 5000

    # Number of features
    p = 50

    # Correlation between each active variable and its paired confounder
    r = 0.3

    # Target FDR
    fdr_target = 0.1

    # Effect size
    es = 1.0

    np.random.seed(1)
    offset = 0
    k = p//2
    nsim_x = 1
    nsim_yx = 1000
    nsim_uyx = 1
    rslt = kosim(nsim_x, nsim_yx, nsim_uyx, n, p, k, r, es, fdr_target,
            offset=offset, corstr='2block',
            betatype='firsthalf', stypes=['equi', 'ldet'], wtypes=['ols','crossprod'], utypes=['util_rand', 'utheta'],
            fixGram=True, center=False, scale=False,
            target_ufrac = p / (n-p))
