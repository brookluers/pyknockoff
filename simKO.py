import knockoff as ko
import genXY as gen
import numpy as np
import scipy
import pandas as pd
import argparse
import sys

np.set_printoptions(precision=5, suppress=True)

sfunc_d = {
    'equi': ko.get_svec_equi,
    'ldet': ko.get_svec_ldet
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
    elif betatype == 'seq':
        brange = 3.3
        bmin = effsize - brange / 2
        bmax = effsize + brange / 2
        beta = gen.rand_beta_seq(p, k, bmin, bmax)
    else:
        print("Unknown betatype, using 'flat'")
        beta = gen.rand_beta_flat(p, k, effsize)
    return beta


def power_method(A, p, startvec, niter=10):
    ek = startvec
    for _ in range(niter):
        ek1 = np.matmul(A, ek)
        ek1_norm = np.linalg.norm(ek1)
        ek = ek1 / ek1_norm
    return np.matmul(np.matmul(ek.T, A), ek)


def kosim(nsim_x, nsim_yx, nsim_uyx, N, p, k, rho,
            effsize, nW=1, FDR=0.1, offset=1, corstr='exch',
            betatype = 'flat',
            stypes = ['equi', 'ldet'],
            wtypes = ['ols', 'crossprod'],
            utype = 'random', bootType='bootThresh',
            scale=True, center=True,
             fixGram = False, to_csv = True, tag='',
             saveW = False, rUUYf = False):
    # Use power iterations to approximate
    # the smallest eigenvalue of X^t X
    # use for log-det and equivariant
    # tuning of S
    start_unit = np.repeat(1/p, p)
    start_unit = start_unit / np.linalg.norm(start_unit)
    beta = get_beta(betatype, p, k, effsize)
    parm_fs = "-N{:d}-p{:d}-k{:d}-rho{:.2f}-nW{:d}"
    parm_fs += "-off{:d}" if bootType != 'multiKO' else "-off{:2.3f}"
    bfn = "beta-" + betatype
    bfn += "-x{:d}-yx{:d}-uyx{:d}-".format(nsim_x,nsim_yx,nsim_uyx) + corstr + parm_fs.format(N, p, k, rho, nW, offset)
    bfn += '-' + tag
    if saveW:
        bfn += '-saveW'
    bfn += '.csv'
    bdf = pd.DataFrame({'beta_j': beta, 'j': np.arange(p)})
    bdf.to_csv(bfn, index = False)
    Sigma = get_Sigma(corstr, p, k, rho)
    ppv = gen.get_ppvfunc(beta)
    tpr = gen.get_tprfunc(beta)
    fdp = gen.get_fdpfunc(beta)
    fpr = gen.get_fprfunc(beta)
    if fixGram:
        print("Generating X with fixed Gram matrix")
        genXfunc = gen.gen_X_given_Gram
    else:
        genXfunc = gen.gen_X
    SigmaChol = np.linalg.cholesky(Sigma)
    if offset == 1:
        print("Knockoff+, offset = {:d}".format(offset))
    elif offset == 0:
        print("Knockoff, offset = {:d}".format(offset))
    else:
        print("Warning: offset is not zero or one, offset = " + str(offset))
    print("true effects = \n\t" + str(beta))
    print("cov(X)[0:5, 0:5]: \t" + str(Sigma[0:5,0:5]).replace('\n','\n\t\t\t'))

    vheader = ['jx','jyx','juyx', 'fdp', 'tpr', 'fpr', 'ppv']
    vheader.extend(["sel{:d}".format(i) for i in range(p)])
    if rUUYf:
        if saveW:
            print("cannot save W when returning ||UUY^2||/||Y||^2, ignoring saveW")
            saveW = False
        if nW == 1:
            vheader.extend(['uyfrac'])
        else:
            print("cannot return ||UUY^2||/||Y||^2 when nW > 1, ignoring rUUYf")
            rUUYf = False
    if saveW:
        if nW == 1 or utype=='split':
            vheader.extend(["W{:d}".format(i) for i in range(p)])
        else:
            print("cannot save W when nW > 1 and not doing split-sample, ignoring saveW")
            saveW = False
    vheader.extend(['wtype_ix', 'stype_ix'])
    vheader.extend(['N','p','k','rho','offset','FDR','nW'])
    gparm = [N, p, k, rho, offset, FDR, nW]
    pddtypes = {'wtype_ix': int, 'stype_ix': int, 'nW': int, 'jx': int, 'jyx': int, 'juyx':int, 'p':int,'k':int,'N':int,
        **{'sel{:d}'.format(i): int for i in range(p)}}
    rslt = []
    sw_ix = [np.array([i,j]) for i in range(len(wtypes)) for j in range(len(stypes))]
    for jx in range(nsim_x):
        X = genXfunc(N, p, SigmaChol, scale, center)
        Qx, Rx = np.linalg.qr(X, mode='reduced')
        G = np.matmul(Rx.T, Rx) # = X^t X
        Ginv = np.linalg.inv(G)
        minEV = 1 / power_method(Ginv, p, startvec = start_unit, niter = 30)
        slist = [sfunc_d[stype](G, minEV = minEV) for stype in stypes]
        cmlist = [ko.get_cmat(X, sv, Ginv) for sv in slist]
        for jyx in range(nsim_yx): # Generate Y | X
            Y = gen.gen_Y(X, N, beta)
            for juyx in range(nsim_uyx):
                # Generate Utilde | X, Y
                selWlist = [  ko.doKnockoff(
                    X, Y, FDR, offset=offset, svec=svec, wstat=wstat,
                    scale=False, center=False,
                    utype=utype, nrep=nW,
                    Qx=Qx, Rx=Rx, Ginv=Ginv, G=G,
                    Cmat=cm, returnW=saveW, bootType=bootType, rUUYf=rUUYf
                    ) for wstat in wtypes for (svec, cm) in zip(slist, cmlist) ]
                if saveW or rUUYf:
                    rslt.extend([np.concatenate((np.array([jx,jyx,juyx,fdp(s), tpr(s), fpr(s), ppv(s)]),1 * s, w, ix)) for ((s, w), ix) in zip(selWlist, sw_ix)])
                else:
                    rslt.extend([np.concatenate((np.array([jx,jyx,juyx,fdp(s), tpr(s), fpr(s), ppv(s)]), 1 * s, ix)) for (s, ix) in zip(selWlist, sw_ix)])
    rslt = np.array(rslt)
    rslt = np.concatenate((rslt, np.broadcast_to(gparm, (rslt.shape[0],len(gparm)))), axis=1)
    df = pd.DataFrame(rslt, columns=vheader)
    df = df.astype(pddtypes)
    df['stype'] = [stypes[si] for si in df['stype_ix'].to_list()]
    df['wtype'] = [wtypes[wi] for wi in df['wtype_ix'].to_list()]
    df = df.drop(columns=['stype_ix','wtype_ix'])
    if to_csv:
        fname = "ko-x{:d}-yx{:d}-uyx{:d}-".format(nsim_x,nsim_yx,nsim_uyx) + corstr + parm_fs.format(N, p,k, rho,nW, offset)
        if betatype != 'flat':
            fname += "-beta" + betatype
        if fixGram:
            fname += '-fixGram' + str(fixGram)
        if saveW:
            fname += '-saveW'
        fname += '-' + tag
        fname += '.csv'
        df.to_csv(fname, index = False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", help="sample size", type=int)
    parser.add_argument("p", help="num. features", type=int)
    parser.add_argument("k", help = "true num. non-null features", type=int)
    parser.add_argument("rho", help="pop. correlation among features",type=float)
    parser.add_argument("nsim_x", help="num. X monte carlo reps",
    type=int)
    parser.add_argument("-nW", help="num. W vectors (knockoff selections) for fixed (X, Y)", type=int, default=1)
    parser.add_argument("-fdr", help="target FDR",
    type=float, default=0.1)
    parser.add_argument("-offset", help="one or zero, for Knockoffs+ or Knockoffs", type=int, default=0)
    parser.add_argument("-corstr", help="correlation structure",
        choices=['ar1','exch'], default='ar1')
    parser.add_argument("-effsize", help="mangitude of true effects",
    type=float, default=3.5)
    parser.add_argument("-nsim_yx", help="num. Y | X monte carlo reps", type=int, default=1)
    parser.add_argument("-nsim_uyx", help="num. (Utilde) | (Y,X) monte carlo reps", type=int, default=1)
    parser.add_argument("-wtype", nargs='+',
            help="importance statistic",
            choices=['crossprod','ols','lasso_coef','lasso_coefIC', 'ridge'],
            default=['crossprod', 'lasso_coefIC'])
    parser.add_argument("-stype", nargs='+',
            help="s_1..s_p tuning",
            choices=['ldet','equi'],
            default=['equi','ldet'])
    parser.add_argument("-utype", help='type of utilde matrix',
            choices=['random','varfrac','runif', 'fixed', 'split'], default='random')
    parser.add_argument('-seed', help='rng seed', type=int)
    parser.add_argument('-saveW', help='save W statistics?',
        type=bool, default=False)
    parser.add_argument('-rUUYf', help='save ||UUY||^2/||Y||^2?',
        type=bool, default=False)
    parser.add_argument('-ftag', help='append to output file name',
        default='')
    parser.add_argument('-btype',help='beta coef pattern',
        default='flat', choices=['flat','first_k','seq'])
    parser.add_argument('-bootType', help='type of bootstrap aggregation of Knockoff selections',
        default='bootThresh', choices=['bootThresh','avg_nsel','multiKO'])
    args = parser.parse_args()
    N, p, k, rho= (args.N, args.p, args.k, args.rho)
    offset, corstr, FDR, es = (args.offset, args.corstr, args.fdr, args.effsize)
    wtypes, stypes, betatype = (args.wtype, args.stype, args.btype)
    nsim_x, nsim_yx, nsim_uyx, nW = (args.nsim_x, args.nsim_yx, args.nsim_uyx, args.nW)
    bootType = args.bootType
    saveW, utype = (args.saveW, args.utype)
    rUUYf = args.rUUYf
    ftag = args.ftag
    if nW > 1 and bootType == 'multiKO':
        print("Gimenez multi-knockoffs, setting offset=1/nW")
        offset = 1 / nW
    else:
        ftag += '-u' + args.utype if args.utype != 'random' else ''
        ftag += '-uuyf' if rUUYf else ''
    ftag += '-' + bootType if nW > 1 and utype != 'split' else ''
    ftag += '-seed' + str(args.seed) if args.seed else ''
    ftag += '-es' + str(args.effsize) if args.effsize else ''
    rng = np.random.default_rng(args.seed)
    ko.rng = rng
    gen.rng = rng
    if nW > 1 and utype =='fixed':
        sys.exit("if utype is fixed, nW must be 1")
    rslt = kosim(nsim_x=nsim_x, nsim_yx=nsim_yx,
                nsim_uyx=nsim_uyx, N=N, p=p, k=k, rho=rho,
                effsize=es, nW=nW, FDR=FDR,
                offset=offset, corstr=corstr,
                betatype = betatype,
                stypes = stypes, wtypes = wtypes,
                utype = utype , bootType=bootType,
                saveW=saveW, tag=ftag, rUUYf = rUUYf)
