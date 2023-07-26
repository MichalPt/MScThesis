import numpy as np
from time import time

from .utils import init_qrh, print_times
from .corfuncs import get_cf_getter
import sys

sys.path.insert(1, '../quantarhei')
import quantarhei as qr


def init_system(parameters, vibrelax=True):
    pp = parameters
    
    # try:
    #     init_qrh(pp)
    # except:
    #     raise Exception("Quantarhei directory not found.")
    
    t0 = time()
    
    with qr.energy_units(pp['units']):
        m1 = qr.Molecule(pp['energies'])

        m1.set_dipole((0,1), pp['dipy'])  # Qy transition
        m1.set_dipole((0,2), pp['dipx'])  # Qx transition

        mod1 = qr.Mode(pp['omega'][0])
        m1.add_Mode(mod1)

        Ng, Ne, Nf = pp['N']
        mod1.set_nmax(0,Ng)  # set number of states in the electronic ground state
        mod1.set_nmax(1,Ne)  #     state 1
        mod1.set_nmax(2,Nf)  #     state 2

        mod1.set_HR(1, pp['hr1'][0])  # Huang-Rhys factor of the mode in state 1
        mod1.set_HR(2, pp['hr1'][1])  # state 2

        if pp['second_mode']:
            mod2 = qr.Mode(pp['omega'][1])
            m1.add_Mode(mod2)
            mod2.set_nmax(0,Ng)
            mod2.set_nmax(1,Ne)
            mod2.set_nmax(2,Nf)
            mod2.set_HR(1, pp['hr2'][0])
            mod2.set_HR(2, pp['hr2'][1])

            #  alpha*Q_1 - beta*Q_2
            alpha = pp['dia_alpha1']
            beta = pp['dia_alpha2']
            m1.set_diabatic_coupling((1, 2), [alpha, [1,0]]) 
            m1.set_diabatic_coupling((1, 2), [beta, [0,1]])
        else:
            # alpha*Q_1
            alpha = pp['dia_alpha1']
            m1.set_diabatic_coupling((1, 2), [alpha, [1]])
    
    """System bath interaction"""
    m1.unset_transition_environment((0,1))
    m1.unset_transition_environment((0,2))


    ta = qr.TimeAxis(int(pp['ti']), int(pp['tf']), pp['dt'])
    
    cfce_el = get_cf_getter('OverdampedBrownian')(ta, pp, pp['spectral_density_el'])
    cfce_vib_11 = get_cf_getter('scaledOverdampedBrownian')(ta, pp, pp['scaled_OB_CF_vib_11'])
    cfce_vib_12 = get_cf_getter('scaledOverdampedBrownian')(ta, pp, pp['scaled_OB_CF_vib_12'])

    m1.set_transition_environment((0,1), cfce_el)
    m1.set_transition_environment((0,2), cfce_el)

    if vibrelax:
        m1.set_mode_environment(0,1,cfce_vib_11)
        m1.set_mode_environment(0,2,cfce_vib_12)

    if pp['second_mode']:
        cfce_vib_21 = get_cf_getter('scaledOverdampedBrownian')(ta, pp, pp['scaled_OB_CF_vib_21'])
        cfce_vib_22 = get_cf_getter('scaledOverdampedBrownian')(ta, pp, pp['scaled_OB_CF_vib_22'])

        if vibrelax:
            m1.set_mode_environment(1,1,cfce_vib_21)
            m1.set_mode_environment(1,2,cfce_vib_22)

    #m1.build()
    
    t1 = time()
    print_times({'System init':t1-t0})
    
    return m1, ta







    
    


    