import numpy as np
from time import time
from datetime import datetime
import sys

from .utils import init_qrh, print_times
from .corfuncs import *

sys.path.insert(1, '../quantarhei')
import quantarhei as qr



def init_system(parameters):
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

    m1._built
    
    """System bath interaction"""
    m1.unset_transition_environment((0,1))
    m1.unset_transition_environment((0,2))
    
    ta = qr.TimeAxis(pp['ti'], pp['tf'], pp['dt'])
    

    # cfs_getters = {'OverdampedBrownian':get_OverdampedBrownian_CF, 
    #                'Wendling':get_Wendling_CF,
    #                'B777':get_B777_CF,
    #                'Reimers2013':get_Reimers2013_CF,
    #               }
    
    # assert pp['spectral_density'] in cfs_getters.keys()
    
    cfce_el = get_cf_getter('OverdampedBrownian')(ta, pp, pp['spectral_density_el'])

    m1.set_transition_environment((0,1), cfce_el)
    m1.set_transition_environment((0,2), cfce_el)
    
    t1 = time()
    print_times({'System init':t1-t0})
    
    return m1, ta


