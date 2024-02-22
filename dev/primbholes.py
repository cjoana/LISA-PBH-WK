import numpy as np
import scipy.constants as const
import scipy.special as special
from scipy.special import erfc
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.integrate import dblquad
import scipy.optimize as opt
import matplotlib.pyplot as plt

import matplotlib as mpl


import sys, os
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
SOURCEPATH = os.path.abspath(os.path.join(ROOTPATH, 'source'))
PARAMSPATH = os.path.abspath(os.path.join(ROOTPATH, 'params'))
PLOTSPATH = os.path.abspath(os.path.join(ROOTPATH, 'plots'))
sys.path.append(ROOTPATH)
sys.path.append(SOURCEPATH)
sys.path.append(PARAMSPATH)


from baseclass import CLASSBase
from power_spectrum import PowerSpectrum
from threshold import ClassThresholds
from merger_rates import MergerRates
from abundances import CLASSabundances



class primbholes(CLASSBase,CLASSabundances,MergerRates):
    def __init__(self, *args, **kwargs):
        # super(CLASSabundances, self).__init__(*args, **kwargs)
        CLASSabundances.__init__(self, *args, **kwargs)
        # MergerRates.__init__(self, *args, **kwargs)

        
        self.use_thermalhistory = False
        pass

    
    def set_powerspectrum(self, model=None, ps_values=None, k_values=None, ps_function=None):
        if ps_function:
            self.ps_function = ps_function
        pass

    def set_thermal_history(self):
        pass

    def get_abundances(self):
        pass

    
    def get_merger_rates(self):
        pass


    






if __name__ == "__main__":

    pb = primbholes()

    
    
    ##Example:
    ## Model A: Gaussian
    sig =  0.25
    As = 0.01*sig
    kp = 1e6
    ps_model = PowerSpectrum.gaussian(As=As, sigma=sig, kp=kp)

    ## Select with vacuum
    ps_func =  ps_model.PS_plus_vacuum        # This is the default to calculate sigma and fPBH

    pb.set_powerspectrum(ps_function=ps_func)

    pb = primbholes(ps_function=ps_func)
    # pb = CLASSabundances(ps_function = ps_func, threshold_method="standard")

    mass = 10**np.linspace(-6,8, 30)  #* Msun
    fpbh = pb.get_fPBH(mass)

    floor = 1e-8
    beta = pb.get_beta(mass)  #+ floor
    fpbh = pb.get_fPBH(mass)  + floor
    sigma = pb.get_variance(mass)

    print(f"fPBH = {fpbh}")


