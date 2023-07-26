import numpy as np
from datetime import datetime
import os, sys
from joblib import Parallel, delayed

from ..utils import bold_text

sys.path.insert(1, '../quantarhei')
import quantarhei as qr


def iterate_parameter(func, parameters, dct, parallel=True, njobs=12, verbose=5, list_of_dicts=True):
    #assert len(dct) == 1, "Only one parameter can be varied at a time"
    
    pars = parameters.copy()
    date_stamp = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
    pars['save_dir'] = pars['save_dir'] + date_stamp
    pars['save'] = True
    dirpath = os.path.join(pars['home_path'], pars['save_dir'])
    print(dirpath)
    
    if parallel:
        if list_of_dicts:
            _iterate_parameter_parallel(func, pars, dct, njobs=njobs, verbose=verbose)
        else:
            _iterate_parameter_parallel_old(func, pars, dct, njobs=njobs, verbose=verbose)
    else:
        _iterate_parameter_serial(func, pars, dct)

    return dirpath


def _iterate_parameter_serial(func, pars, dct):
    par, vals = list(dct.items())[0]
    length = len(vals)
    
    for i, val in enumerate(vals):
        print(bold_text(">>> Value")+" {}/{} ...".format(i+1,length))
        pars.update({par:val})
        func(pars)
        
    print(bold_text('Done!'))


def _iterate_parameter_parallel_old(func, pars, dct, njobs=12, verbose=9):
    par, vals = list(dct.items())[0]    
        
    def run_calc(f, ps, p, v):
        ps['use_gpu'] = False
        ps.update({p:v})
        f(ps)

    Parallel(n_jobs=njobs, verbose=verbose)(delayed(run_calc)(func, pars, par, val) for val in vals)
  
    print(bold_text('Done!'))


def _iterate_parameter_parallel(func, pars, dct, njobs=1, verbose=9):
    def run_calc(f, ps, p):
        ps['use_gpu'] = False
        print(p)
        ps.update(p)
        f(ps)

    Parallel(n_jobs=njobs, verbose=verbose)(delayed(run_calc)(func, pars, p) for p in dct)
  
    print(bold_text('Done!'))


def align_spects(alignto, labels, xaxis, spects):
    get_max_x = lambda i: xaxis[i][np.argwhere(spects[i] == np.max(spects[i]))[0]]
    
    if type(alignto) == str:
        assert alignto in labels
        index = labels.index(alignto)
        maxat = get_max_x(index)
    elif (type(alignto) == int) or (type(alignto) == float):
        maxat = alignto
        index = -1
    else:
        raise Exception('Unsupported type!')
        
    new_xaxes = list()

    for i, (x, y) in enumerate(zip(xaxis, spects)):
        if i == index:
            new_xaxes.append(x)
            continue

        mat = get_max_x(i)
        new_xaxes.append(x - mat + maxat)

    return new_xaxes, spects


def reorder_labels(N, C):
    NC = int(N/C)
    indices = np.arange(N)
    xindices = indices[:NC*C].reshape(NC, C)
    yindices = indices[NC*C:]
    
    x0indices = xindices[:,:len(yindices)]
    x1indices = xindices[:,len(yindices):]

    return np.append(np.vstack([x0indices, yindices]).T.reshape((1,-1)), x1indices.T.reshape((1,-1)))


def fit_exponentials(xdata, ydata, n=4, return_data=False, print_relax_times=True):
    ### https://math.stackexchange.com/questions/1428566/fit-sum-of-exponentials/3808325#3808325
    ### https://www.scribd.com/doc/14674814/Regressions-et-equations-integrales
    
    from scipy import integrate
    
    def calc_integral(xarr, yarr, n=1):
        out = yarr
        
        for i in range(n):
            out = integrate.cumulative_trapezoid(out, xarr, initial=0) 
        
        return out
    
    xx = np.array([xdata**i for i in range(n)])
    yy = np.array([calc_integral(xdata, ydata, n=i+1) for i in range(n)])
    
    Y = np.vstack([yy, xx[::-1]])
    A = np.dot(np.transpose(np.linalg.pinv(Y), [1,0]), ydata)
    Abar = np.eye(n, k=-1)
    Abar[0,:] = A[:n]
    _lambdas, dmat = np.linalg.eig(Abar)
    lambdas = np.concatenate([[0], _lambdas])
    
    newY = np.array([np.exp(lamb * xdata) for lamb in lambdas])
    P = np.dot(np.transpose(np.linalg.pinv(newY), [1,0]), ydata)

    data = np.sum(np.array([a*np.exp(b*xdata) for a,b in zip(P, lambdas)]), axis=0)

    output = [P, -lambdas]

    if print_relax_times:
        print('\n'.join(['{}: {:.0f} fs'.format(i,np.real(num)) for i, num in enumerate(np.sort(1 / lambdas[1:]))]))
    
    if return_data:
        output.append(np.real(data))
    
    return output


def pulse_width_from_duration(time, time_units='fs'):
    from scipy.constants import value

    time_units_dct = {'s':1.0, 'ns':1e9, 'ps':1e12, 'fs':1e15}
    assert time_units in time_units_dct.keys()
    
    freq = 1e-2 * time_units_dct[time_units] / (value('speed of light in vacuum') * time)
    return freq


