import numpy as np
import sys

sys.path.insert(1, '../quantarhei')
import quantarhei as qr

def get_cf_getter(name):
    cfs_getters = {'OverdampedBrownian':get_OverdampedBrownian_CF, 
                   'Wendling':get_Wendling_CF,
                   'B777':get_B777_CF,
                   'Reimers2013':get_Reimers2013_CF,
                   'scaledOverdampedBrownian':get_scaled_OverdampedBrownian_CF,
                   'mixed_CF':get_mixed_CF,
                  }
        
    assert name in cfs_getters.keys()

    return cfs_getters[name]

def join_pars(pars1, pars2, keys):
    out = dict()
    
    for key in keys:
        if key in pars2.keys():
            out[key] = pars2[key]
        elif key in pars1.keys():
            out[key] = pars1[key]
        else:
            raise Exception('Unknown key encountered in the list of keys')
            
    return out


def get_OverdampedBrownian_CF(timeax, pars, pp=None):
    if pp is not None:
        pars = join_pars(pars, pp, ['reorg', 'cortime', 'T', 'matsubara', 'units'])
        
    with qr.energy_units(pars['units']):
        cf = qr.CorrelationFunction(timeax, dict(ftype="OverdampedBrownian", 
                                                 reorg=pars['reorg'], cortime=pars['cortime'],
                                                 T=pars['T'], matsubara=pars['matsubara']))
    return cf


def get_Wendling_CF(timeax, pars, pp=None):
    from quantarhei.models.spectral_densities.wendling_2000 import wendling_2000a
    
    if pp is not None:
        pars = join_pars(pars, pp, ['T', ])
        
    wendling = wendling_2000a()
    spd = wendling.get_SpectralDensity(timeax)
    cf = spd.get_CorrelationFunction(pars['T'])
    return cf


def get_B777_CF(timeax, pars, pp=None):
    if pp is not None:
        pars = join_pars(pars, pp, ['reorg', 'cortime', 'T', 'matsubara', 'gamma', 'B777_alternative_form', 'units'])
        
    with qr.energy_units(pars['units']):
        cf = qr.CorrelationFunction(timeax, dict(ftype="B777", reorg=pars['reorg'], gamma=pars['gamma'], 
                                    cortime=pars['cortime'], T=pars['T'], matsubara=pars['matsubara'],
                                    alternative_form=['B777_alternative_form']))
    return cf


def get_Reimers2013_CF(timeax, pars, pp=None):
    from quantarhei.models.spectral_densities.reimers_2013 import Reimers_2013

    if pp is not None:
        pars = join_pars(pars, pp, ['T', ])
        
    reimers = Reimers_2013()
    spd = reimers.get_SpectralDensity(timeax)
    cf = spd.get_CorrelationFunction(pars['T'])
    return cf


def get_scaled_OverdampedBrownian_CF(timeax, pars, pp=None):
    from quantarhei.qm.corfunctions.correlationfunctions import oscillator_scaled_CorrelationFunction
    
    if pp is not None:
        pars = join_pars(pars, pp, ['reorg', 'cortime', 'T', 'matsubara','omega', 'HR', 'Nmax', 'target_time','units'])
    
    with qr.energy_units(pars['units']):
        #print('CTime ', pars['cortime'])
        cf = oscillator_scaled_CorrelationFunction(timeax, 
                                                   dict(ftype="OverdampedBrownian", 
                                                        reorg=pars['reorg'], cortime=pars['cortime'],
                                                        T=pars['T'], matsubara=pars['matsubara']),
                                                   omega=pars['omega'], target_time=pars['target_time'], 
                                                   Nmax=pars['Nmax'], HR=pars['HR'], silent=True)
        print('Reorg:',cf.measure_reorganization_energy())
    return cf

def get_mixed_CF(timeax, pars):
    mixed_pars = pars['mixed_CF']
    cfs = mixed_pars['cfs']
    coeffs = mixed_pars['coeffs']
    assert len(cfs) == len(coeffs), "The numbers of provided CFs and weight coefficients differ!"
    assert 'mixed_CF' not in cfs, "Back-reference detected. You can't mix mixed CF with itself."

    with qr.energy_units('1/cm'):
        cf_fin = get_cf_getter(cfs[0])(timeax, pars)
        data = cf_fin.data * coeffs[0]
        
        for cf, coeff in zip(cfs[1:], coeffs[1:]):
            data += get_cf_getter(cf)(timeax, pars).data * coeff

    print(data.shape, cf_fin.data.shape)
    cf_fin.data = data
    
    return cf_fin
    

def plot_correlation_function(system, plotto=None, return_data=False, **kwargs):
    import matplotlib.pyplot as plt

    H = system.get_Hamiltonian()
    cf = system.get_SystemBathInteraction().CC
    xdata = cf.timeAxis.data
    ydata = cf.data[1]

    if plotto is None:
        plotto = plt
    else:
        pass

    plotto.plot(xdata, ydata, **kwargs)

    if return_data:
        return xdata, ydata

    