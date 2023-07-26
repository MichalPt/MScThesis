import matplotlib.pyplot as plt
import numpy as np

def plt_latex():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size = 11)
    plt.rc('text.latex', preamble=r'\usepackage{mathpazo} \usepackage{tgpagella} \usepackage{upgreek} \usepackage{amsmath}')


def ftex(string):
    return r'{}'.format(string)


def plt_color():
    from cycler import cycler

    ccolor = cycler('color', plt.get_cmap('Dark2').colors)
    plt.rcParams['axes.prop_cycle'] = ccolor

def get_color_linestyle_cycler():
    from cycler import cycler
    from itertools import cycle
    
    color_cycle = cycler('color', plt.get_cmap('Dark2').colors)
    ls_cycle = cycler('ls', ['-',':', '-.'])
    
    cyc = ls_cycle * color_cycle
    return cycle(cyc)


def format_ticks(axes):
    pars = dict(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                bottom=True,           top=True,      left=True,      right=True,
               )
    
    if type(axes) == list:
        for a in axes:
            a.tick_params(**pars)
    else:
        axes.tick_params(**pars)


def matplotlib_legend():
    from matplotlib.legend_handler import HandlerLine2D
    from matplotlib.lines import Line2D

    def update_prop(handle, orig):
        handle.update_from(orig)
        #handle.set(marker='s')
        #handle.set(linestyle='')
        handle.set_linewidth(5)
        
    handler_pars = dict(update_func=update_prop, xpad=1.5)
    handler = HandlerLine2D(**handler_pars)
    handler_map = {Line2D: handler}
    
    #return handler_map
    return dict(handler_map=handler_map, labelspacing=0, columnspacing=1, handletextpad=0.6)
    #return dict(labelspacing=0, columnspacing=1.2, handletextpad=0.02)


def get_relative_pos(values, x):
    if x in values:
        out = values.index(x)
    else:
        for i, (y0, y1) in enumerate(zip(values[:-1], values[1:])):
            if x > y0 and x < y1:
                out = (x - y0) / (y1 - y0) + i

    return out

def get_abs_pos(values, xid):
    x0 = int(xid)
    x1 = x0 + 1

    val0 = values[x0]
    val1 = values[x1]
    val = (val1 - val0) * (xid - x0) + val0
    return val

def format_cbar_ticks(values, min_sep=None, labels=None):
    
    n = len(values)

    if min_sep is None:
        min_sep = values[-1] - values[-2]

    old_positions = np.linspace(values[0], values[-1], n)
    positions = list()
    
    if labels is None:
        labels = list()
    
        positions.append(values[0])
        labels.append(values[0])
        
        for i, (val, old_pos) in enumerate(zip(values[1:], old_positions[1:])):
    
            if (val - labels[-1]) >= min_sep:
                labels.append(val)
                positions.append(old_pos)
         
    else:
        positions = [get_abs_pos(old_positions, get_relative_pos(list(labels), lab)) for lab in labels]

    return positions, labels