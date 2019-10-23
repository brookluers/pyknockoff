import numpy as np
import knockoff as ko
import genXY as gen

if __name__ == "__main__":
    N = 5000

    # Number of features
    p = 100

    # Number of nonzero variables
    k = 50

    # Correlation
    rho = 0.3
    # Populaiton covariance matrix
    Sigma = gen.get_ar(p, rho)
    SigmaChol = np.linalg.cholesky(Sigma)

    # Target FDR
    FDR = 0.1
    offset = 0
    # Effect size
    es = 3.5
    nsim_x = 2
    nsim_yx = 3
    nW = 10
    wtype = 'crossprod'

    np.random.seed(1)
    beta = gen.rand_beta_flat(p, k, es)
    fdp = gen.get_fdpfunc(beta)
    tpr = gen.get_tprfunc(beta)

    fname = "ko-wvote-nx" + str(nsim_x) + "-nyx" + str(nsim_yx)
    fname += '-N' + str(N) + '-p' + str(p)
    fname += '-w' + wtype + '-nW'+ str(nW) + '-ar1-rho' + str(rho) + '-off' + str(offset)
    fname += '.csv'

    rslt = []
    vheader = ['jx','jyx','nW','fdp','tpr']
    vheader.extend(["sel{:d}".format(i) for i in range(p)])
    vfmt = ['%d', '%d', '%d', '%.18e', '%.18e']
    vfmt.extend(['%d' for i in range(p)])
    vheader = ','.join(vheader)
    for jx in range(nsim_x):
        X = gen.gen_X(N, p, SigmaChol) # scaled and centere
        Qx, Rx = np.linalg.qr(X, mode='reduced')
        G = np.matmul(Rx.T, Rx)
        svec = ko.get_svec_ldet(G)
        Ginv = np.linalg.inv(G)
        Cmat = ko.get_cmat(X, svec, Ginv=Ginv)
        for jyx in range(nsim_yx):
            Y = gen.gen_Y(X, N, beta, sigma=1)
            selmat = []
            for i in range(nW):
                selmat.append(
                    ko.doKnockoff(X, Y, FDR, offset, svec=svec,
                            wstat=wtype, scale=False, center=False,
                            Qx = Qx, Rx=Rx, Ginv=Ginv,
                            Cmat=Cmat)
                    )
            selmat = np.array(selmat)
            avg_nsel = int(np.round(np.mean(np.sum(selmat, axis=1))))
            ranksel = np.argsort(np.sum(selmat, axis=0))
            sel_consensus = np.repeat(False, p)
            sel_consensus[ranksel[-avg_nsel:]] = True
            sel_one = selmat[0, :]
            r1 = [jx, jyx, 1, fdp(sel_one), tpr(sel_one)]
            r1.extend((1 * sel_one).tolist())
            rc = [jx, jyx, nW, fdp(sel_consensus), tpr(sel_consensus)]
            rc.extend((1 * sel_consensus).tolist())
            rslt.append(r1)
            rslt.append(rc)
    rslt = np.array(rslt)
    np.savetxt(fname, rslt,fmt=vfmt, delimiter=',', header=vheader, comments='')
