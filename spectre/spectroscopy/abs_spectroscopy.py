import numpy as np
import matplotlib.pyplot as plt
from time import time
from datetime import datetime
import sys, os
from matplotlib.ticker import MultipleLocator

from ..utils import print_times, check_path, save2file, save2json, load_binary, get_differences_in_dicts, bold_text
from .spec_utils import align_spects

sys.path.insert(1, '../../quantarhei')
import quantarhei as qr
from quantarhei.qm.hilbertspace.operators import ReducedDensityMatrix
from quantarhei.core.diskarray import DiskArray
from spectre.matplotlib_latex import *
from spectre.spectroscopy.spec_utils import reorder_labels
from cycler import cycler
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import contextlib

plt_latex()
plt_color()

def get_propagator(system, timeax, parameters):
    pp = parameters

    prop = system.get_ReducedDensityMatrixPropagator(timeax, relaxation_theory=pp['relaxation_theory'],
                                                     secular_relaxation=pp['secular_relaxation'], 
                                                     time_dependent=pp['time_dependent'], 
                                                     use_gpu=False, #pp['use_gpu'], 
                                                     method=pp['method'], 
                                                     my_prop=pp['my_propagator'],
                                                     as_operators=pp['time_dependent'],
                                                     use_diskarray=pp['use_diskarray'])
    return prop


def calculate_abs_spectrum(system, timeax, parameters, return_propagator=False):
    pp = parameters
    # if (pp['time_dependent'] == True) and (pp['secular_relaxation'] == False) and (pp['use_gpu'] == True):
    #     pp['use_gpu'] = False
    
    # if (pp['use_gpu'] == False):
    #     pp['parallel'] = True
        
    hardware = {True:'GPU', False:'CPU'}
    
    t0 = time()
    prop = get_propagator(system, timeax, pp)
    t1 = time()
    
    with qr.energy_units(pp['units']):
        system.HH.set_rwa([0, 1])
        dia = system.HH.data.diagonal()
        rwa = (dia[None,:]-dia[:,None])[0,1:].mean()
        absc = qr.AbsSpectrumCalculator(timeax, system)
        absc.bootstrap(rwa=rwa, prop=prop)
        spect = absc.calculate(from_dynamics=True, 
                               truncate_system=pp['truncate_system'], truncate_lims=pp['truncate_lims'])
                               #parallel=pp['parallel'], njobs=pp['njobs'], alt=pp['alternative'])
    
    t2 = time()    
    
    print_times({'Tensor calculation':t1-t0, "Spectrum calculation ({})".format(hardware[pp['use_gpu']]):t2-t1})
    
    #spect.normalize()
    if pp['save']:
        directory = os.path.join(pp['home_path'], pp['save_dir'])
        file = check_path(os.path.join(directory, "abs_spectrum.bin"))
        pp['file'] = file
        pp['timedate'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        t3 = time()
        if not os.path.exists(directory):
            os.mkdir(directory)
        save2file([spect, prop, pp], file)
        save2json(pp, file.replace('.bin','.json'))
        t4 = time()
        
        print_times({'Saving':t4-t3})
        
    if return_propagator:
        return spect, prop
    else:
        return spect


def plot_multi_spectra(directory=None, paths=None, units='1/cm', datalim=None,
                       figsize=(6,5), dpi=300, xlim=(14000,20000), ylim=(-0.03,1.05), 
                       alignat=None, return_data=False, colormap='viridis',
                       ncols=None, use_as_labels=None, sort=True, reversed=False, 
                       normalize=True, yshift=None,
                       **kwargs):

    assert os.path.isdir(directory) or type(paths) == list
    
    spect_data = list()
    xaxis = list()
    pars = list()
    
    if os.path.isdir(directory):
        files = [file for file in os.listdir(directory) if file.endswith('.bin') and file.startswith('abs')]
        for file in files:
            data = load_binary(os.path.join(directory,file))
            spect = data[0]
            pars.append(data[2])
                
            with qr.energy_units(units):
                if normalize:
                    spect.normalize()
                spect_data.append(spect.data)
                xaxis.append(spect.axis.data)
         
    else:
        for fpath in paths:
            if os.path.isfile(fpath):
                data = load_binary(fpath)
                spect = data[0]
                pars.append(data[2])

                with qr.energy_units(units):
                    if normalize:
                        spect.normalize()
                    spect_data.append(spect.data)
                    xaxis.append(spect.axis.data)
    
    key, labs = get_differences_in_dicts(*pars, use_as_labels=use_as_labels)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    from matplotlib import cm

    if alignat is not None:
        xaxis, spect_data = align_spects(alignat, labs, xaxis, spect_data)
    
    zipped = list(zip(xaxis, spect_data, labs))
    
    if datalim is not None:
        zipped = [(x,y,l) for (x,y,l) in zipped if l >= datalim[0] and l <= datalim[1]]
    
    clm = cm.get_cmap(colormap, len(list(zipped)))
    
    if sort:
        zipped = sorted(zipped, key = lambda x: x[2])
        
    if reversed:
        zipped = zipped[::-1]
    
    dy = 0
    labformat = '{}'                  
    
    for i, (xdata, ydata, label) in enumerate(zipped):
        if type(label) in [list, tuple]:
            label = str(label)
            
        elif type(label) in [float, int]:
            if float(label) < 1:
                labformat = '{:.03f}'
        else:
            pass
        
        plt.plot(xdata, ydata + dy, label=labformat.format(label), color=clm(i), **kwargs)
        
        if yshift is not None:
            dy += yshift
    
    ll = len(zipped)
    if ncols == None:
        if ll < 6:
            ncols = ll
        else:
            ncols = 6
    
    plt.title(key, y=1.06 + 0.05*int(ll / ncols), fontsize='medium' )
    plt.legend(frameon=False, fontsize='small', bbox_to_anchor=(0.5, 1.08 + 0.05*int(ll / ncols)), ncols=ncols, loc='upper center')
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel(r'wavenumber [{}]'.format(units))
    plt.ylabel('absorbance [$a.u.$]')
    #plt.show()
    
    if return_data:
        return xaxis, spect_data, labs


def plot_ref(indices=None, dir_path="C:/Users/micha/Documents/Studium/MScThesis/exp-abs-spectra", 
             alignat=None, prnt=False, plot=True, plot_to=None, return_data=False, 
             crop=True, crop_threshold=0.01, labels=False, 
             return_spline=False, spline_s=0.0002, spline_k=3, **kwargs):
    
    files = {0:'BChla-glycerol-water-5K-1-Zazubovich-2001.npy',
             1:'BChla-TEA-5K-1-Zazubovich-2001.npy',
             2:'Chla-diethylether-295K-1-Ratsep-2009.npy',
             3:'Chla-propan-1-ol-295K-1-Ratsep-2009.npy',
             4:'Chla-propan-2-ol-295K-1-Ratsep-2009.npy',
             5:'Chla-THF-295K-1-Ratsep-2009.npy',
             6:'Chla-pyridine-295K-1-Ratsep-2009.npy',
             7:'Chla-propan-1-ol-4K-1-Ratsep-2009.npy',
             8:'Chla-propan-2-ol-4K-1-Ratsep-2009.npy',
             9:'BChla-ethylether-1-Reimers-2013.npy',
             10:'BChla-pyridine-1-Reimers-2013.npy',
             11:'Chla-ethylether-1-Reimers-2013.npy',
             12:'Chla-pyridine-1-Reimers-2013.npy',
             13:'Chla-ethylether-2-Reimers-2013.npy',
             14:'Chla-MeOH-EtOH-1-Reimers-2013.npy',
             15:'Chla-propan-1-ol-2K-1-Reimers-2013.npy',
             16:'Chla-propan-1-ol-160K-1-Reimers-2013.npy',
             17:'Chla-propan-2-ol-2K-1-Reimers-2013.npy',
             18:'BChla-TEA-4K-1-Ratsep-2019.npy',
             19:'Chla-TEA-4K-2-Ratsep-2019.npy',
            }
                 
    if prnt:
        #print(files)
        return files
    
    def plot_graph(fp, alignto=alignat, plot=plot, kwargs=kwargs):
        fpath = os.path.join(dir_path, fp)
        data = np.load(fpath)

        if labels:
            kwargs['label'] = fp.split('.',1)[0]

        if crop:
            min_index = int(np.argwhere(data[1] >= crop_threshold)[0])
            max_index = int(np.argwhere(data[1] >= crop_threshold)[-1])
            data = [d[min_index:max_index] for d in data]
        
        if alignto == None:
            pass
        elif type(alignto) is int or type(alignto) is float:
            maxat = data[0][np.argwhere(data[1] == np.max(data[1]))[0]]
            data = [data[0] - maxat + alignto, data[1]]
        else:
            raise Exception('Unsupported type')
                            
        if plot is True:
            if plot_to is not None:
                pltto = plot_to
            else:
                pltto = plt
                
            pltto.plot(*data, **kwargs)
            
        return np.array(data)

                 
    if indices == None:
        data = list()
        for f in files.values():
            idata = plot_graph(f)
            data.append(idata)
    
    elif type(indices) == int:
        data = plot_graph(files[indices])
        
    elif type(indices) == list:
        data = list()
        for i in indices:
            idata = plot_graph(files[i])
            data.append(idata)
            
    elif type(indices) == str:
        data = plot_graph(indices)
    
    else:
        raise Exception('Unsupported type')

    if return_data == 'spline':
        from scipy.interpolate import UnivariateSpline
        
        if type(data[0]) == list:
            spline = list()
            for (x, y) in data:
                ix = np.linspace(x[0], x[-1], 20000)
                ispline = UnivariateSpline(x,y, k=spline_k, s=spline_s)
                spline.append([ix, ispl])
        else:
            x = data[0]
            ix = np.linspace(x[0], x[-1], 20000)
            spline = [ix, UnivariateSpline(*data, k=spline_k, s=spline_s)]
            
        return spline
        
    elif return_data == 'data' or return_data is True:
        return data


def plot_single_spectrum(specdata, syst, pars, show_dmoments=True, units='1/cm', return_ax=True, **kwargs):
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    
    with qr.energy_units('1/cm'):
        ydata = specdata.data
        xdata = specdata.axis.data
        
    fig, ax = plt.subplots(nrows=2, figsize=(6,4.5), dpi=300, height_ratios=(1,0.2),sharex=True)
    ax[0].plot(xdata, ydata, **kwargs)
    ax[1].fill(xdata, ydata, **kwargs, facecolor='grey', edgecolor='none',alpha=0.15)
    
    with qr.energy_units('1/cm'):
        ev, mat = np.linalg.eigh(syst.HH.data)
        
    trdm = np.transpose(syst.get_TransitionDipoleMoment().data, [2,1,0])

    if pars['second_mode']:
        N = np.array(pars['N'])**2
    else:
        N = np.array(pars['N'])

    filter_x = np.zeros(np.sum(N))
    filter_y = np.zeros(np.sum(N))
    filter_x[N[0]+N[1]:] = 1
    filter_y[N[0]:-N[2]] = 1

    filter_xx = np.ones_like(filter_x)[:,np.newaxis] * filter_x
    filter_yy = np.ones_like(filter_y)[:,np.newaxis] * filter_y

    tr = np.einsum('ij,njk,kl->nil',np.linalg.inv(mat), trdm, mat)
    tr_x = np.einsum('ij,njk,kl->nil',np.linalg.inv(mat), trdm*filter_xx, mat)
    tr_y = np.einsum('ij,njk,kl->nil',np.linalg.inv(mat), trdm*filter_yy, mat)

    dty = np.sum(tr_y**2, axis=0)[0,1:]
    dtx = np.sum(tr_x**2, axis=0)[0,1:]
    dt = np.sum(tr**2, axis=0)[0,1:]

    dt = dt / np.max(dt)
    ratio = dt / (dtx + dty)
    
    path = Path([[10000, 0], [10000, 1], [25000, 1], [25000, 0]])
    patch = PathPatch(path, facecolor='none',edgecolor='none')
    ax[1].add_patch(patch)    
    
    for x,y,e in zip(dtx*ratio, dty*ratio, ev[1:]):
        para = dict(linewidth=2.4, clip_path=patch, clip_on=True)
        ax[1].plot([e, e], [0, y], c='r', label=r'Q$_y$', **para)
        ax[1].plot([e, e], [y, y+x], c='b', label=r'Q$_x$', **para)
    
    # remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax[1].legend(by_label.values(), by_label.keys(), frameon=False, loc='center right')
    
    plt.subplots_adjust(hspace=.0, wspace=0.0)
    plt.setp(ax[0].get_xticklabels(), visible=False)
    plt.setp(ax[0].get_xticklines(), visible=False)
    ax[1].set_ylim(-0.1,1.1)
    ax[1].set_xlabel(r'$\omega$ [1/cm]')
    ax[0].set_ylabel(r'$\alpha$($\omega$) [a.u.]')
    ax[1].set_ylabel(r'rel. $d^2$')
    
    ax[1].plot([10000,25000],[0,0],linewidth=0.1,c='k')

    if return_ax:
        return ax

    
def plot_abs_and_dynamics(system, specdata, prop, pars, basis='energy', show_dmoments=True, return_ax=True, xlim=(14000,20000), 
                          clip_dynamics=True, ncols=8, print_coeffs=True, **kwargs):

    H = system.get_Hamiltonian()
    
    ### INITIAL RHO

    fax, dat = get_exc_pulse(system, pars)
    spect = qr.DFunction(fax, dat)
    
    populations = calculate_dynamics(system, None, pars, basis=basis, propagator=prop, exc_pulse=spect)
                
    ### ABSORPTION SPECTRUM
    
    with qr.energy_units(pars['units']):
        ydata = specdata.data
        xdata = specdata.axis.data
        xdata2 = fax.data

        
    ### PLOT ALLL
    #   -------------
    #   |     3     |
    #   -------------
    #   |     4     |
    #   -------------
    #   | 0/1 | 0/2 |
    #   -------------

    fig = plt.figure(figsize=(6,7), dpi=300,)
    outer_grid = fig.add_gridspec(2, 1, hspace=0.23, height_ratios=(1,0.7))
    ax = list()
    
    ax.append(fig.add_subplot(outer_grid[1, 0]))
    
    inner_grid_0 = outer_grid[1, 0].subgridspec(ncols=2, nrows=1, hspace=0, width_ratios=(1,0.4))    # lower graph
    inner_grid_1 = outer_grid[0, 0].subgridspec(ncols=1, nrows=2, hspace=0, height_ratios=(1,0.35))  # upper graph
    ax = ax + list(inner_grid_0.subplots(sharey=True)) + list(inner_grid_1.subplots(sharex=True))
    
    ax[3].plot(xdata, ydata, **kwargs)
    ax[4].fill(xdata2, dat, **kwargs, facecolor='0.8', edgecolor='none', alpha=1)
    #ax[4].fill(xdata, ydata, **kwargs, facecolor='0.65', edgecolor='none', alpha=1)
    ax[4].plot(xdata, ydata, **kwargs, color='k', linewidth=0.7, alpha=1)
    
    with qr.energy_units(pars['units']):
        ev, mat = np.linalg.eigh(H.data)

    handle1, handle2 = plot_bars(system, pars, plot_to=ax[4], xlim=xlim, basis=basis, print_coeffs=print_coeffs)
    
    ax[4].legend(handles=[handle1[0], handle2[0]], frameon=False, loc='center right', #handler_map=matplotlib_legend())
                 **matplotlib_legend())
    
    format_ticks(ax)
    
    plt.subplots_adjust(hspace=.0, wspace=0.0)
    ax[2].tick_params(labelleft=False, left=False)
    ax[3].tick_params(labelbottom=False, bottom=False)
    ax[4].tick_params(top=False)

    # AX 0 - behind AX 1+2
    ax[0].set_xlabel(r'time [fs]', labelpad=22)
    ax[0].set_frame_on(False)
    ax[0].tick_params(labelleft=False, labelbottom=False, left=False, bottom=False, right=False, top=False)
    
    # AX 1 - fast dynamics
    ax[1].set_xlim(0,200)
    ax[1].set_ylim(-0.09, 1.09)
    ax[1].set_ylabel(r'populations')
    #ax[1].set_xlabel(r'time [fs]')
    ax[1].xaxis.set_minor_locator(MultipleLocator(25))

    # AX 2 - slow dynamics
    ax[2].set_xlim(200,2000)
    #ax[2].set_xlabel(r'time [ps]')
    ax[2].xaxis.set_minor_locator(MultipleLocator(200))
    
    # AX 3+4 - spectra
    ax[3].set_ylim(-0.07, 1.07)
    ax[4].set_xlim(*xlim)
    ax[4].set_ylim(-0.32,1.3)
    ax[4].ticklabel_format(axis='x',style='scientific',scilimits=(0,2))
    ax[4].set_xlabel(r'$\omega$ [cm$^{-1}$]')
    ax[3].set_ylabel(r'$\alpha$($\omega$) [a.u.]')
    ax[4].set_ylabel(r'rel. $d^2$')
    ax[4].xaxis.set_minor_locator(MultipleLocator(500))
    
    ax[4].plot([10000,25000], [0,0], linewidth=0.1, c='k')
    
    cyc = get_color_linestyle_cycler()

    ### DYNAMICS
    for i, pop in enumerate(populations[1:]):
        if ev[i] > xlim[1] and clip_dynamics:
            continue
        linepars = {**next(cyc), 'linewidth':1.2}
        i += 1
        ax[1].plot(prop.TimeAxis.data, pop, label = str(i), marker='', **linepars)
        ax[2].plot(prop.TimeAxis.data, pop, label = str(i), marker='', **linepars)

    handles, labels = ax[1].get_legend_handles_labels()

    indices = reorder_labels(len(handles), ncols)
    handles = [handles[i] for i in indices]
    labels = [labels[i] for i in indices]

    ax[0].legend(handles, labels, frameon=False, loc='upper center', ncols=ncols, bbox_to_anchor=(0.45,-0.2),
                 **matplotlib_legend())
                              
    if return_ax:
        return ax

def plot_abs_and_dynamics_poster(system, specdata, prop, pars, basis='energy', show_dmoments=True, return_ax=True, xlim=(14000,20000), 
                          clip_dynamics=True, ncols=8, print_coeffs=True, **kwargs):

    H = system.get_Hamiltonian()
    
    ### INITIAL RHO

    fax, dat = get_exc_pulse(system, pars)
    spect = qr.DFunction(fax, dat)
    
    populations = calculate_dynamics(system, None, pars, basis=basis, propagator=prop, exc_pulse=spect)
                
    ### ABSORPTION SPECTRUM
    
    with qr.energy_units(pars['units']):
        ydata = specdata.data
        xdata = specdata.axis.data
        xdata2 = fax.data

        
    ### PLOT ALLL
    #   -------------
    #   |     3     |
    #   -------------
    #   |     4     |
    #   -------------
    #   | 0/1 | 0/2 |
    #   -------------

    fig = plt.figure(figsize=(6,7), dpi=300,)
    outer_grid = fig.add_gridspec(2, 1, hspace=0.23, height_ratios=(1,0.7))
    ax = list()
    
    ax.append(fig.add_subplot(outer_grid[1, 0]))
    
    inner_grid_0 = outer_grid[1, 0].subgridspec(ncols=2, nrows=1, hspace=0, width_ratios=(1,0.4))    # lower graph
    inner_grid_1 = outer_grid[0, 0].subgridspec(ncols=1, nrows=2, hspace=0, height_ratios=(1,0.35))  # upper graph
    ax = ax + list(inner_grid_0.subplots(sharey=True)) + list(inner_grid_1.subplots(sharex=True))
    
    ax[3].plot(xdata, ydata, **kwargs, linewidth=2)
    #ax[4].fill(xdata2, dat, **kwargs, facecolor='0.8', edgecolor='none', alpha=1)
    #ax[4].fill(xdata, ydata, **kwargs, facecolor='0.65', edgecolor='none', alpha=1)
    ax[4].plot(xdata, ydata, **kwargs, color='k', linewidth=0.7, alpha=1)
    
    with qr.energy_units(pars['units']):
        ev, mat = np.linalg.eigh(H.data)

    handle1, handle2 = plot_bars(system, pars, plot_to=ax[4], xlim=xlim, basis=basis, print_coeffs=print_coeffs, bar_width=3.1)
    
    ax[4].legend(handles=[handle1[0], handle2[0]], frameon=False, loc='center right', #handler_map=matplotlib_legend())
                 **matplotlib_legend())
    
    format_ticks(ax)
    
    plt.subplots_adjust(hspace=.0, wspace=0.0)
    ax[2].tick_params(labelleft=False, left=False)
    ax[3].tick_params(labelbottom=False, bottom=False)
    ax[4].tick_params(top=False)

    # AX 0 - behind AX 1+2
    ax[0].set_xlabel(r'time [fs]', labelpad=22)
    ax[0].set_frame_on(False)
    ax[0].tick_params(labelleft=False, labelbottom=False, left=False, bottom=False, right=False, top=False)
    
    # AX 1 - fast dynamics
    ax[1].set_xlim(0,200)
    ax[1].set_ylim(-0.09, 1.09)
    ax[1].set_ylabel(r'populations')
    #ax[1].set_xlabel(r'time [fs]')
    ax[1].xaxis.set_minor_locator(MultipleLocator(25))

    # AX 2 - slow dynamics
    ax[2].set_xlim(200,2000)
    #ax[2].set_xlabel(r'time [ps]')
    ax[2].xaxis.set_minor_locator(MultipleLocator(200))
    
    # AX 3+4 - spectra
    ax[3].set_ylim(-0.07, 1.07)
    ax[4].set_xlim(*xlim)
    ax[4].set_ylim(-0.32,1.3)
    ax[4].ticklabel_format(axis='x',style='scientific',scilimits=(0,2))
    ax[4].set_xlabel(r'$\omega$ [cm$^{-1}$]')
    ax[3].set_ylabel(r'$\alpha$($\omega$) [a.u.]')
    ax[4].set_ylabel(r'rel. $d^2$')
    ax[4].xaxis.set_minor_locator(MultipleLocator(500))
    
    ax[4].plot([10000,25000], [0,0], linewidth=0.4, c='k')
    
    cyc = get_color_linestyle_cycler()

    ### DYNAMICS
    for i, pop in enumerate(populations[1:]):
        if ev[i] > xlim[1] and clip_dynamics:
            continue
        elif ev[i] > 19500:
            continue
            
        linepars = {**next(cyc), 'linewidth':1.2}
        #i += 1
        ax[1].plot(prop.TimeAxis.data, pop, label = str(i), marker='', **linepars)
        ax[2].plot(prop.TimeAxis.data, pop, label = str(i), marker='', **linepars)

    handles, labels = ax[1].get_legend_handles_labels()

    indices = reorder_labels(len(handles), ncols)
    handles = [handles[i] for i in indices]
    labels = [labels[i] for i in indices]

    ax[0].legend(handles, labels, frameon=False, loc='upper center', ncols=ncols, bbox_to_anchor=(0.45,-0.2),
                 **matplotlib_legend())
                              
    if return_ax:
        return ax


def calculate_dynamics(system, timeax, pars, propagator=None, basis='energy', exc_pulse=None, return_rho=False):
    pp = pars
    H = system.get_Hamiltonian()
    contexts = {'local':contextlib.nullcontext(), 'energy':qr.eigenbasis_of(H)}
    assert basis in contexts.keys()
    
    ### INITIAL RHO

    if exc_pulse is None:
        pulse_freq = pars['exc_pulse_freq']
        pulse_width = pars['exc_pulse_width']
        
        with qr.eigenbasis_of(H):
            om1 = 10000
            dom = 1.0
            No = 20000
            dat = np.zeros(No, dtype=qr.REAL)
            
            with qr.energy_units(pars['units']):
                fax = qr.FrequencyAxis(om1, No, dom)
            
                for io in range(No):
                    om = fax.data[io]
                    dat[io] = np.exp(-((om-pulse_freq) / pulse_width)**2)
                    
                # normalize the pulse
                ssum = np.sum(dat)*dom
                dat = dat/ssum
                mx = np.max(dat)
                dat = dat/mx
                
                spect = qr.DFunction(fax, dat)
    else:
        spect = exc_pulse
        fax = spect.axis
    
    rhoi = system.get_excited_density_matrix(condition=["pulse_spectrum", spect])
    rhoi.normalize()

    if propagator is None:
        propagator = get_propagator(system, timeax, pp)
        
    ### DYNAMICS
    
    rhot = propagator.propagate(rhoi)
    
    if not isinstance(rhot.data, DiskArray):
        with contexts[basis]:
            populations = np.einsum('tii->it', rhot.data).copy()
    else:
        shp = rhot.data.shape
        populations = np.zeros((shp[1], shp[0]), dtype=float)
        
        for tt in range(shp[0]):
            dmat = np.array(rhot.data[tt,:,:])
            rdmat = ReducedDensityMatrix(data=dmat)
            with contexts[basis]:
                populations[:,tt] = np.real(np.diagonal(rdmat.data))

    if pars['save']:
        directory = os.path.join(pars['home_path'], pars['save_dir'])
        file = check_path(os.path.join(directory, "dynamics.bin"))
        pars['file'] = file
        pars['timedate'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        if not os.path.exists(directory):
            os.mkdir(directory)
            
        save2file([populations, rhot, pars], file)
        save2json(pars, file.replace('.bin','.json'))

    if return_rho:
        with contexts[basis]:
            return rhot.data
    else:
        return np.real(populations)


def calculate_bars_old(system, pars):
    H = system.get_Hamiltonian()
    
    with qr.energy_units(pars['units']):
        ev, dmat = np.linalg.eigh(H.data)
        
    trdm = np.transpose(system.get_TransitionDipoleMoment().data, [2,1,0])

    if pars['second_mode']:
        N = np.array(pars['N'])**2
    else:
        N = np.array(pars['N'])

    filter_qx = np.zeros(np.sum(N))
    filter_qy = np.zeros(np.sum(N))
    filter_qx[N[0]+N[1]:] = 1
    filter_qy[N[0]:-N[2]] = 1

    filter_xx = np.ones_like(filter_qx)[:,np.newaxis] * filter_qx
    filter_yy = np.ones_like(filter_qy)[:,np.newaxis] * filter_qy

    tr = np.einsum('ij,njk,kl->nil', np.linalg.inv(dmat), trdm, dmat)
    tr_qx = np.einsum('ij,njk,kl->nil', np.linalg.inv(dmat), trdm*filter_xx, dmat)
    tr_qy = np.einsum('ij,njk,kl->nil', np.linalg.inv(dmat), trdm*filter_yy, dmat)

    dty = np.sum(tr_qy**2, axis=0)[0,1:]
    dtx = np.sum(tr_qx**2, axis=0)[0,1:]
    dt = np.sum(tr**2, axis=0)[0,1:]

    dt = dt / np.max(dt)
    ratio = dt / (dtx + dty)

    return dtx*ratio, dty*ratio, ev[1:]

def calculate_bars(system, pars, print_coeffs=True):
    x = pars['dipx']
    y = pars['dipy']
    
    if pars['second_mode']:
        N = np.array(pars['N'])**2
    else:
        N = np.array(pars['N'])

    H = system.get_Hamiltonian()
    trdm = system.get_TransitionDipoleMoment().data
    
    with qr.energy_units(pars['units']):    
        ev, dmat = np.linalg.eigh(H.data)
    
    def get_invmatrix(x, y):
        z = np.cross(x,y)
        M = np.vstack([x,y,z]).T
        Minv = np.linalg.inv(M)
        return Minv
    
    coeffs = np.einsum('mn,ij,jkn,kl->mil', get_invmatrix(x,y), np.linalg.inv(dmat), trdm, dmat)[:,0,1:]
    out = coeffs**2 / np.sum(coeffs**2, axis=0)
    tr = np.einsum('ij,jkn,kl->nil', np.linalg.inv(dmat), trdm, dmat)

    dt = np.sum(tr**2, axis=0)[0,1:]
    norm = np.ones_like(dt) #dt / np.max(dt)

    if print_coeffs:
        print('\n'+bold_text('Transition dipole moment coefficients:')+'\n ID         dE     c1x    c2y    c3z  |  c1x^2  c2y^2  c3z^2  |    d^2')
        print('----------------------------------------------------------------------')
        for i, (e, (x,y,z), d2) in enumerate(zip(ev[1:], coeffs.T, norm)):
            print(' {:>2.0f}: {:>9.2f}   {:>5.2f}  {:>5.2f}  {:>5.2f}  |   {:>4.2f}   {:>4.2f}   {:>4.2f}  |  {:>5.3f}'.format(i+1,e-ev[0],x,y,z,x**2,y**2,z**2,d2))
        print('')
        
    return out[0] * norm, out[1] * norm, ev[1:] - ev[0]
    

def plot_bars(system, pars, plot_to=None, xlim=(14000,20000), basis='energy', bars_labs_yshift=-0.25, print_coeffs=True, bar_width=2.4):
    if plot_to is None:
        plot_to = plt

    path = Path([[xlim[0], 0], [xlim[0], 1], [xlim[1], 1], [xlim[1], 0]])
                                  
    patch = PathPatch(path, facecolor='none', edgecolor='none',clip_on=True)
    plot_to.add_patch(patch)    
    nn = 1

    ### BARS
    bars = calculate_bars(system, pars, print_coeffs=print_coeffs)
    #bars = calculate_bars_old(system, pars)
    
    for x,y,e in zip(*bars):
        para = dict(linewidth=bar_width, clip_path=patch, clip_on=True, solid_capstyle='butt')

        if e > xlim[1]:
            continue
        elif e > 19100:
            continue
                    
        # handle1 = plot_to.plot([e, e], [0, y], c='r', label=r'Q$_y$', **para)
        # handle2 = plot_to.plot([e, e], [y, y+x], c='b', label=r'Q$_x$', **para)
        handle1 = plot_to.plot([e, e], [0, x], c='b', label=r'Q$_x$', **para)
        handle2 = plot_to.plot([e, e], [x, y+x], c='r', label=r'Q$_y$', **para)
        
        if basis == 'energy':
            plot_to.text(e, bars_labs_yshift, '{}'.format(nn), ha='center', fontsize=8, clip_on=True)
            pass
        nn += 1

    return handle1, handle2


def plot_residuum(spectra, ref_id, plot_to=None, legend=True, dir_path="C:/Users/micha/Documents/Studium/MScThesis/exp-abs-spectra"):
    refspline = plot_ref(ref_id, linestyle='dashed', c='k', linewidth=0.8, return_data='spline', label='experiment', plot=False, dir_path=dir_path)
    
    if plot_to is None:
        plot_to = plt

    with qr.energy_units('1/cm'):
        ydata = spectra.data
        xdata = spectra.axis.data
        resid = ydata - refspline[1](xdata)
        xycut = [[x,y] for x,y in zip(xdata, resid) if (x > refspline[0][0]) and (x < refspline[0][-1])]
        xcut, ycut = np.array(xycut).T
        plot_to.plot(xcut, ycut, linewidth=0.6, color='0.75', linestyle='-', label='residuum')
        plot_to.fill_between(xcut, ycut, linewidth=0.6, color='0.93', linestyle='')
        
        if legend:
            plot_to.legend(frameon=False)
        

def get_exc_pulse(syst, pars):
    pulse_freq = pars['exc_pulse_freq']
    pulse_width = pars['exc_pulse_width']
    H = syst.get_Hamiltonian()
    
    with qr.eigenbasis_of(H):
        om1 = 10000
        dom = 1.0
        No = 20000
        dat = np.zeros(No, dtype=qr.REAL)
        
        with qr.energy_units(pars['units']):
            fax = qr.FrequencyAxis(om1, No, dom)
            shift = H.data[0,0]
        
            for io in range(No):
                om = fax.data[io]
                dat[io] = np.exp(-((om-pulse_freq) / pulse_width)**2)
                
            # normalize the pulse
            ssum = np.sum(dat)*dom
            dat = dat/ssum
            mx = np.max(dat)
            dat = dat/mx
            
            # spect = qr.DFunction(fax, dat)
    return fax, dat
    