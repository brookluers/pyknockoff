import knockoff as ko
import genXY as gen
import numpy as np
import scipy
import pandas as pd

np.set_printoptions(precision=5, suppress=True)

wfunc_d = {
    'ols': lambda X, Xk, Y, G2p, cp2p: ko.stat_ols(X, Xk, Y, G2p, cp2p),
    'crossprod': lambda X, Xk, Y, G2p, cp2p:  ko.stat_crossprod(X, Xk, Y, cp2p=cp2p),
    'lasso_coef': lambda X, Xk, Y, G2p, cp2p: ko.stat_lasso_coef(X, Xk, Y, precompute=G2p),
    'lasso_coefIC': lambda X, Xk, Y, G2p, cp2p: ko.stat_lassoLarsIC_coef(X, Xk, Y, precompute=G2p, criterion='bic', max_iter=100)
}

theta_seq = np.linspace((1/4)*np.pi, (3/4)*np.pi, 150)

utfunc_d = {
    'util_rand': lambda Qx, N, p, Y, Rx, ut1=None: ko.get_util_random(Qx, N, p) if ut1 is None else ut1,
    'utheta': lambda Qx, N, p, Y, Rx, ut1: ko.get_utheta_fixfrac(Qx, N, p, Y, Rx, tseq=theta_seq, ut1 = ut1)
}

sfunc_d = {
    'equi': lambda G, minEV: ko.get_svec_equi(G, minEV = minEV),
    'ldet': lambda G, minEV: ko.get_svec_ldet(G, minEV = minEV)
}

def get_Sigma(corstr, p, k, rho):
    if corstr == 'exch':
        print("Exchangeable covariance matrix")
        Sigma = gen.get_exch(p, rho)
    elif corstr == '2block':
        print("Using '2block' covariance matrix")
        Sigma = gen.get_2block(p, k, rho)
    elif corstr == 'ar1':
        print("Using AR covariance")
        Sigma = gen.get_ar(p, rho)
    else:
        print("Unknown corstr, using exchangeable")
        Sigma = gen.get_exch(p, rho)
    return Sigma

def get_beta(betatype, p, k, effsize):
    if betatype == 'flat':
        beta = gen.rand_beta_flat(p, k, effsize)
    elif betatype == 'first_k':
        beta = gen.fix_beta_first_k(p, k, effsize)
    else:
        print("Unknown betatype, using 'flat'")
        beta = gen.rand_beta_flat(p, k, effsize)
    return beta

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
        ek1 = np.matmul(A, ek)
        ek1_norm = np.linalg.norm(ek1)
        ek = ek1 / ek1_norm
    return np.matmul(np.matmul(ek.T, A), ek)


def kosim(nsim_x, nsim_yx, nsim_uyx, N, p, k, rho,
            effsize, FDR=0.1, offset=1, corstr='exch', betatype='flat', stypes = ['equi', 'ldet'],
            wtypes = ['ols', 'crossprod'],
            utypes = ['utheta', 'util_rand'],
            scale=True, center=True,
             fixGram = False, to_csv = True):
    simparm_global = {
        'N': N, 'p': p, 'k': k, 'FDR': FDR,
        'rho': rho, 'corstr': corstr,
        'offset': offset
    }
    nstypes = len(stypes)
    nutypes = len(utypes)
    # Use power iterations to approximate
    # the smallest eigenvalue of X^t X
    # use for log-det and equivariant
    # tuning of S
    rand_unit = np.random.normal(size=(p,))
    rand_unit = rand_unit / np.linalg.norm(rand_unit)
    beta = get_beta(betatype, p, k, effsize)
    Sigma = get_Sigma(corstr, p, k, rho)
    ppv = get_ppvfunc(beta)
    tpr = get_tprfunc(beta)
    fdp = get_fdpfunc(beta)
    fpr = get_fprfunc(beta)
    if fixGram:
        print("Generating X with fixed Gram matrix")
        genXfunc = gen.gen_X_given_Gram
    else:
        genXfunc = gen.gen_X
    SigmaChol = np.linalg.cholesky(Sigma)
    genYfunc = lambda X, N: gen.gen_Y(X, N, beta)
    if offset == 1:
        print("Knockoff+, offset = {:d}".format(offset))
    elif offset == 0:
        print("Knockoff, offset = {:d}".format(offset))
    else:
        print("offset must be 0 or 1, setting to 1")
        offset = 1
    print("true effects = \n\t" + str(beta))
    print("cov(X)[0:5, 0:5]: \t" + str(Sigma[0:5,0:5]).replace('\n','\n\t\t\t'))
    rslt_keys = ['ppv','tpr','fdp','fpr','nsel']
    rslt_keys.extend(["sel{:d}".format(i) for i in range(p)])
    rslt = []
    for jx in range(nsim_x):
        X = genXfunc(N, p, SigmaChol, scale, center)
        Qx, Rx = scipy.linalg.qr(X, mode='economic')
        ut1 = ko.get_util_random(Qx, N, p)
        G = np.matmul(Rx.T, Rx) # = X^t X
        Ginv = scipy.linalg.inv(G)
        minEV = 1 / power_method(Ginv, p, startvec = rand_unit, niter = 30)
        slist = [sfunc_d[stype](G, minEV) for stype in stypes]
        cmlist = [ko.get_cmat(X, sv, Ginv) for sv in slist]
        for jyx in range(nsim_yx): # Generate Y | X
            Y = genYfunc(X, N)
            XYcp = np.matmul(X.T, Y)
            for juyx in range(nsim_uyx): # Generate Utilde | X, Y
                # (X, Y) are fixed
                # one Utilde for each utype
                Utlist = [utfunc_d[utype](Qx, N, p, Y, Rx, ut1) for utype in utypes]
                ufracs = [np.sum(np.matmul(Ut, np.matmul(Ut.T, Y))**2) / np.sum(Y**2) for Ut in Utlist]
                # len(utypes) * len(stypes)
                Xtlist = [ko.getknockoffs_qr(X, G, sv_r, Qx, N, p, Ut, Ginv, cm_r) for Ut in Utlist for (sv_r, cm_r) in zip(slist, cmlist)]
                # pre-compute np.matmul(X.T, Xk)
                X_Xk_cplist = [np.matmul(X.T, Xk) for Xk in Xtlist]
                Xk_Y_cplist = [np.matmul(Xk.T, Y) for Xk in Xtlist]
                # List of (  2p-dim Gram Matrix, (X Xtilde)^t Y )
                #   for each Xtilde
                suffStat = [ (np.block([[G, XkX], [XkX,G]]),
                                np.concatenate((XYcp, Xk_Ycp))) for
                                (Xk_Ycp, XkX) in zip(Xk_Y_cplist, X_Xk_cplist)]
                # len(wtypes) * len(Xtlist)
                #    = len(wtypes) * len(utypes) * len(stypes)
                Wlist = [wfunc_d[wtype](X, Xk, Y, G2p, cp2p) for wtype in wtypes for (Xk, (G2p, cp2p)) in zip(Xtlist, suffStat)]
                simparm_inner_byW = [
                        {'wtype': wtype, 'utype': utype, 'ufrac': uf, 'stype': stype, 'juyx': juyx, 'jyx': jyx, 'jx': jx}
                    for wtype in wtypes for (utype, uf) in zip(utypes, ufracs) for stype in stypes]
                sel_byW = [(wvec >= ko.knockoff_threshold(wvec, FDR, offset)).tolist() for wvec in Wlist]
                res_byW = [{**dict(zip(rslt_keys,
                                        [ppv(sel), tpr(sel), fdp(sel), fpr(sel), sum(sel)] + [1 * sj for sj in sel ])), **pj}
                            for (sel, pj) in
                            zip(sel_byW, simparm_inner_byW)]
                rslt.extend(res_byW)

    df = pd.DataFrame(rslt)
    for pk in simparm_global:
        df[pk] = simparm_global[pk]
    if to_csv:
        df.to_csv("ko-x{:d}-yx{:d}-uyx{:d}-".format(nsim_x,nsim_yx,nsim_uyx) + corstr + "-" + betatype + "-N{:d}-p{:d}-rho{:.2f}-off{:d}-fixGram".format(N, p, rho, offset) + str(fixGram) + ".csv", index=False)
    return df


if __name__ == "__main__":

    # Sample size
    n = 3000

    # Number of features
    p = 100

    # Correlation between each active variable and its paired confounder
    r = 0.9
    # Target FDR
    fdr_target = 0.1

    # Effect size
    es = 3.5

    np.random.seed(1)
    offset = 0
    k = 20
    nsim_x = 10
    nsim_yx = 1
    nsim_uyx = 1
    rslt = kosim(nsim_x, nsim_yx, nsim_uyx, n, p, k, r, es, fdr_target,
            offset=offset, corstr='exch',
            betatype='flat', stypes=['equi', 'ldet'],
            wtypes=['crossprod', 'ols', 'lasso_coefIC'],
            utypes=['util_rand'],
            fixGram=False, center=True, scale=True)
