import simKnockoff as sko
import numpy as np
import multiprocessing

def parSim_withseed(arg_dict):
    s = arg_dict.pop('seed')
    print(arg_dict)
    np.random.seed(s)
    print("\n---------\n")
    print("\trandom seed: {:f}".format(s))
    sko.kosim(**arg_dict)

if __name__ == "__main__":

    # Sample size
    N = 5000

    # Number of features
    p = 40

    # Test sequence of correlation values
    rhotest = [0.2, 0.4] #, 0.6, 0.8]

    # population correlation structures to test
    corstr_test = ['exch'] # , '2block']
    betatype_test = ['flat'] # , 'firsthalf']

    # Target FDR
    fdr_target = 0.1

    # Effect size
    es = 3.5

    offsets = [0]
    k = p//2
    nsim_x = 10
    nsim_yx = 1
    nsim_uyx = 1
    parlist = [{'nsim_x': nsim_x,
      'nsim_yx': nsim_yx,
      'nsim_uyx': nsim_uyx,
      'N': N, 'p': p,
      'k': k, 'rho': rho,
      'effsize': es,
      'FDR': fdr_target,
      'offset': os,
      'corstr': cs, 'betatype': bt,
      'stypes': ['equi','ldet'],
      'wtypes': ['crossprod', 'lasso_coef'],
      'utypes': ['utheta','util_rand']
    } for rho in rhotest for (cs, bt) in zip(corstr_test, betatype_test) for os in offsets]
    for j in range(len(parlist)):
        parlist[j].update({'seed': j})
    pool = multiprocessing.Pool(processes = multiprocessing.cpu_count()-1)

    rslt = pool.map(parSim_withseed, parlist)
