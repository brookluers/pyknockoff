import bh
import genXY as gen
import numpy as np
import scipy
import pandas as pd
import argparse


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

def bhsim(nsim_x, nsim_yx, N, p, k, rho,
            effsize, nboot = 1, FDR = 0.1, corstr='exch',
            betatype = 'flat',
            scale=True, center=True,
             fixGram = False, to_csv = True, tag='',
             rng=None):
    if rng is None:
        print("No rng passed to simulation, creating non-seeded rng")
        rng = np.random.default_rng()
    print("Benjamini-Hochberg simulation, Guassian linear model\n")
    beta = get_beta(betatype, p, k, effsize)
    bfn = "beta-bh-" + betatype
    bfn += "-x{:d}-yx{:d}-".format(nsim_x,nsim_yx) + corstr
    bfn += "-N{:d}-p{:d}-k{:d}-rho{:.2f}".format(N, p, k, rho)
    if nboot > 1:
        bfn += '-nboot{:d}'.format(nboot)
    if len(tag) > 0:
        bfn += '-' + tag
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
    print("true effects = \n\t" + str(beta))
    print("cov(X)[0:5, 0:5]: \t" + str(Sigma[0:5,0:5]).replace('\n','\n\t\t\t'))
    vheader = ['jx','jyx', 'fdp', 'tpr', 'fpr', 'ppv']
    vheader.extend(["sel{:d}".format(i) for i in range(p)])
    vheader.extend(['N','p','k','rho','FDR','nboot'])
    gparm = [N, p, k, rho, FDR, nboot]
    pddtypes = {'nboot': int, 'jx': int, 'jyx': int, 'p':int,'k':int,'N':int,
        **{'sel{:d}'.format(i): int for i in range(p)}}
    rslt = []
    for jx in range(nsim_x):
        X = genXfunc(N, p, SigmaChol, scale, center)
        print("X^t X[0:5, 0:5] = " + str(np.matmul(X.T,X)[0:5,0:5]).replace('\n','\n\t\t\t'))
        for jyx in range(nsim_yx): # Generate Y | X
            Y = gen.gen_Y(X, N, beta)
            sel = bh.bhOLSReg(X, Y, FDR, nboot=nboot)
            rslt.extend([np.concatenate((np.array([jx,jyx,fdp(sel), tpr(sel), fpr(sel), ppv(sel)]), 1 * sel))])
    rslt = np.array(rslt)
    rslt = np.concatenate((rslt, np.broadcast_to(gparm, (rslt.shape[0],len(gparm)))), axis=1)
    df = pd.DataFrame(rslt, columns=vheader)
    df = df.astype(pddtypes)
    if to_csv:
        fname = "bh-x{:d}-yx{:d}-".format(nsim_x,nsim_yx) + corstr
        fname += "-N{:d}-p{:d}-k{:d}-rho{:.2f}".format(N, p,k, rho)
        if nboot > 1:
            fname += '-nboot{:d}'.format(nboot)
        if betatype != 'flat':
            fname += "-beta" + betatype
        if fixGram:
            fname += '-fixGram' + str(fixGram)
        if len(tag) > 0:
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
    parser.add_argument("-nboot", help="num. bootstrap samples", type=int, default=1)
    parser.add_argument("-fdr", help="target FDR",
    type=float, default=0.1)
    parser.add_argument("-corstr", help="correlation structure",
        choices=['ar1','exch'], default='ar1')
    parser.add_argument("-effsize", help="mangitude of true effects",
    type=float, default=3.5)
    parser.add_argument("-nsim_yx", help="num. Y | X monte carlo reps", type=int, default=1)
    parser.add_argument('-fixGram', help='fix X^t X',
        type=bool, default=False)
    parser.add_argument('-seed', help='rng seed', type=int)
    parser.add_argument('-ftag', help='append to output file name',
        default='')
    parser.add_argument('-btype',help='beta coef pattern',
        default='flat', choices=['flat','first_k','seq'])
    args = parser.parse_args()
    N, p, k, rho= (args.N, args.p, args.k, args.rho)
    fixGram = args.fixGram
    corstr, FDR, es = (args.corstr, args.fdr, args.effsize)
    betatype = args.btype
    nsim_x, nsim_yx, nboot = (args.nsim_x, args.nsim_yx, args.nboot)
    ftag = args.ftag
    ftag += '-seed' + str(args.seed) if args.seed else ''
    ftag += '-es' + str(es) if args.effsize else ''
    rng = np.random.default_rng(args.seed)
    bh.rng = rng
    gen.rng = rng

    rslt = bhsim(nsim_x=nsim_x, nsim_yx=nsim_yx,
                N=N, p=p, k=k, rho=rho,
                effsize=es, nboot=nboot, FDR=FDR, corstr=corstr,
                betatype = betatype, tag=ftag, fixGram = fixGram, rng=rng)
