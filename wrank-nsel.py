import numpy as np
import knockoff as ko
import genXY as gen
import argparse

def sim_wrank(N, p, k, rho, corstr, FDR, offset, nW,
              nsim_x, nsim_yx, nsim_uyx, wtype, toCSV=True, tag=''):
    if corstr == 'ar1':
        Sigma = gen.get_ar(p, rho)
    else:
        Sigma = gen.get_exch(p, rho)
    SigmaChol = np.linalg.cholesky(Sigma)
    beta = gen.rand_beta_flat(p, k, es)
    fdp = gen.get_fdpfunc(beta)
    tpr = gen.get_tprfunc(beta)
    fname = "wrank-" + tag + "-nx" + str(nsim_x) + "-nyx" + str(nsim_yx)
    fname += "-nuyx" + str(nsim_uyx)
    fname += '-N' + str(N) + '-p' + str(p)
    fname += '-w' + wtype + '-nW'+ str(nW) + '-ar1-rho' + str(rho) + '-off' + str(offset)
    fname += '.csv'
    rslt = []
    vheader = ['k','N','p','rho','offset','nW', 'jx','jyx','juyx','FDR', 'fdp','tpr']
    vheader.extend(["sel{:d}".format(i) for i in range(p)])
    vfmt = ['%d', '%d', '%d','%.3f','%d','%d','%d','%d','%d','%.3f', '%.18e', '%.18e']
    vfmt.extend(['%d' for i in range(p)])
    vheader = ','.join(vheader)
    gparm = [k, N, p, rho, offset]
    for jx in range(nsim_x):
        X = gen.gen_X(N, p, SigmaChol) # scaled and centere
        Qx, Rx = np.linalg.qr(X, mode='reduced')
        G = np.matmul(Rx.T, Rx)
        svec = ko.get_svec_ldet(G)
        Ginv = np.linalg.inv(G)
        Cmat = ko.get_cmat(X, svec, Ginv=Ginv)
        for jyx in range(nsim_yx):
            Y = gen.gen_Y(X, N, beta, sigma=1)
            for juyx in range(nsim_uyx):
                sel_consensus = ko.doKnockoff(X, Y, FDR, offset,
                                            svec=svec, wstat=wtype,
                                            scale=False, center=False,
                                            Utilde=None, nrep=nW,
                                            Qx=Qx,Rx=Rx, Ginv=Ginv,G=G,
                                            Cmat=Cmat)
                sel_one = ko.doKnockoff(X, Y, FDR, offset,
                                            svec=svec, wstat=wtype,
                                            scale=False, center=False,
                                            Utilde=None, nrep=1,
                                            Qx=Qx,Rx=Rx,Ginv=Ginv,G=G,
                                            Cmat=Cmat)
                r1 = gparm + [1, jx, jyx, juyx, FDR, fdp(sel_one), tpr(sel_one)]
                r1.extend((1 * sel_one).tolist())
                rc = gparm + [nW, jx, jyx, juyx, FDR, fdp(sel_consensus), tpr(sel_consensus)]
                rc.extend((1 * sel_consensus).tolist())
                rslt.append(r1)
                rslt.append(rc)
    rslt = np.array(rslt)
    if toCSV:
        np.savetxt(fname, rslt,fmt=vfmt, delimiter=',', header=vheader, comments='')
        return rslt
    else:
        return rslt, vheader, vfmt, fname

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", help="sample size", type=int)
    parser.add_argument("p", help="num. features", type=int)
    parser.add_argument("k", help = "true num. non-null features", type=int)
    parser.add_argument("rho", help="pop. correlation among features",type=float)
    parser.add_argument("nW", help="num. W vectors (knockoff selections) for fixed (X, Y)", type=int)
    parser.add_argument("nsim_x", help="num. X monte carlo reps",
    type=int)
    parser.add_argument("-fdr", help="target FDR",
    type=float, default=0.1)
    parser.add_argument("-offset", help="one or zero, for Knockoffs+ or Knockoffs", type=int, default=0)
    parser.add_argument("-corstr", help="ar1 or exch, correlation structure", default='ar1')
    parser.add_argument("-effsize", help="mangitude of true effects",
    type=float, default=3.5)
    parser.add_argument("-nsim_yx", help="num. Y | X monte carlo reps", type=int, default=1)
    parser.add_argument("-nsim_uyx", help="num. (Utilde) | (Y,X) monte carlo reps", type=int, default=1)
    parser.add_argument("-wtype", help="importance statistic, either crossprod, ols, lasso_coef, or lasso_coefIC",
        default='crossprod')
    parser.add_argument('-seed', help='rng seed', type=int)
    args = parser.parse_args()

    N, p, k, rho= (args.N, args.p, args.k, args.rho)
    offset, corstr, FDR, es = (args.offset, args.corstr, args.fdr, args.effsize)
    wtype = args.wtype
    nsim_x, nsim_yx, nsim_uyx, nW = (args.nsim_x, args.nsim_yx, args.nsim_uyx, args.nW)
    # Seed the RNG, if provided
    rng = np.random.default_rng(args.seed)
    ko.rng = rng
    gen.rng = rng
    ftag = 'seed' + str(args.seed) if args.seed else ''

    sim_wrank(N,p,k,rho,corstr,FDR,offset,nW,nsim_x,nsim_yx,nsim_uyx,wtype, tag=ftag)
