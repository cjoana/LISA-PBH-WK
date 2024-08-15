"""
Code to evaluate merger rates given a mass distribution f(m1, m2, z), with PBH masses m1, m2 at a redshift z. 
Basic models are included and others can be easily expanded. 


Need testing!!! 
"""

""" TODO: 
    X 'primodrial binaries' to be renamed as 'early binaries'. ( consistency with paper)
    X 'cluster binaries' to be renamed as 'late binaries'. ( consistency with paper)
    * 'disrupted early binaries' should be considered
    * supression factor, fsub = (S1 * S2) in EB,  should be computed as a function of fpbh, m1 and m2

    X fromula EB : WRONG, missing a factor
    X formula LB:  WRONG, completely different as in paper. Eq. 4.9, in page 53. 


"""


import numpy as np
import scipy.constants as const
import scipy.special as special
from scipy.special import erfc
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.integrate import dblquad
import scipy.optimize as opt

import sys, os
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
SOURCEPATH = os.path.abspath(os.path.join(ROOTPATH, 'source'))
PARAMSPATH = os.path.abspath(os.path.join(ROOTPATH, 'params'))
PLOTSPATH = os.path.abspath(os.path.join(ROOTPATH, 'plots'))
sys.path.append(ROOTPATH)
sys.path.append(SOURCEPATH)
sys.path.append(PARAMSPATH)


# import abundances
from abundances import CLASSabundances

# from user_params import cosmo_params, physics_units
from params.user_params import physics_units, cosmo_params, PSModels_params
from params.user_params import Thresholds_params, MerginRates_params
from params.user_params import verbose 



class MergerRates():    
    def __init__(self, masses=None, fpbhs=None, fpbh_integrated=None):   
        self.Rclust = 400               # TODO : hardcode
        self.fsup = 0.0025              # TODO : It assumes a given value, but it sould depend on fpbh, m1, m2. 
        self.ratio_mPBH_over_mH = 0.8   # TODO : hardcode

        self.masses = masses
        self.fpbhs = fpbhs
        self.fpbh_integrated = fpbh_integrated

        self.Omc = cosmo_params.Omc
        self.Omb = cosmo_params.Omb


        # output
        self.sol_rates_early_binaries = None
        self.sol_rates_late_binaries = None        


    def rates_early_binaries(self, m1, m2, fm1, fm2, fpbh_total):   #TODO : primordial should be renamed ad Early Binaries

        # assuming fidutial suppression_factor
        suppression_factor = self.fsup

        # assuming masses in Msun. 
        rates = 1.6e6  * suppression_factor * fpbh_total**(53/37) * fm1 * fm2   * \
                    (m1 + m2) ** (-32/37) * (m1 * m2 / (m1 + m2)**2) ** (-34/37)

        return rates

    def get_rates_early_binaries(self, fpbh_total=1, masses=None, fpbhs=None):
        
        if isinstance(fpbh_total, bool):
            fpbh_total = self.fpbh_integrated
            if not fpbh_total: raise ValueError("Value for the fpbh_total is not set. (get_rates_early_binaries)")

        if isinstance(masses, bool):
            masses = self.masses
            if not masses: raise ValueError("Values for the masses are not set. (get_rates_early_binaries)")
        if isinstance(fpbhs, bool):
            fpbhs = self.fpbhs
            if not fpbhs: raise ValueError("Values for the fpbhs are not set. (get_rates_early_binaries)")

        # Computes the merging rates of primordial binaries
        Nmass = len(masses)
        rates = np.zeros([Nmass, Nmass]) * np.nan
        for ii in range(Nmass):
            m1 = masses[ii]
            fpbh1 = fpbhs[ii]
            for jj in range(ii):
                m2 = masses[jj]
                fpbh2 = fpbhs[jj]
                rates[ii,jj] = self.rates_early_binaries(m1, m2, fpbh1, fpbh2, fpbh_total)
        self.sol_rates_early_binaries = rates
        return rates


    def rates_late_binaries(self, m1, m2, fm1, fm2, fpbh_total):    #TODO: Formula wrong, or inconsistent with paper p.53
        
        # Assuming fidtual Rclust (typically btw 100 and 1000).
        Rclust = self.Rclust

        # Assuming no suppression factors from late binaries
        rates = Rclust * fpbh_total**2 * fm1 * fm2  * \
                    (m1 + m2)**(10/7) * (m1 * m2)** (-5/7)
                    # (m1 + m2) ** (-32. / 37.) * (m1 * m2 / (m1 + m2) ** 2) ** (-34. / 37.)

        return rates

    def get_rates_late_binaries(self, fpbh_total=None, masses=None, fpbhs=None):

        if isinstance(fpbh_total, bool):
            fpbh_total = self.fpbh_integrated
            if not fpbh_total: raise ValueError("Value for the fpbh_total is not set. (get_rates_late_binaries)")
        if isinstance(masses, bool):
            masses = self.masses
            if not masses: raise ValueError("Values for the masses are not set. (get_rates_late_binaries)")
        if isinstance(fpbhs, bool):
            fpbhs = self.fpbhs
            if not fpbhs: raise ValueError("Values for the fpbhs are not set. (get_rates_late_binaries)")

        # Computes the merging rates for tidal capture in PBH clusters
        Nmass = len(masses) 
        rates = np.zeros([Nmass, Nmass]) * np.nan
        for ii in range(Nmass):
            m1 = masses[ii]
            fpbh1 = fpbhs[ii]
            for jj in range(ii):
                m2 = masses[jj]
                fpbh2 = fpbhs[jj]
                rates[ii,jj] = self.rates_late_binaries(m1, m2, fpbh1, fpbh2, fpbh_total)
        self.sol_rates_late_binaries = rates
        return rates

    # def eval_oldcode(self):
    #     # Compute the merging rates
    #     print("Step 3:  Computation of the PBH merging rates")
    #     print("====")
    #     self.sol_rates_early_binaries = np.zeros((self.Nmass, self.Nmass))
    #     self.sol_rates_cluster = np.zeros((self.Nmass, self.Nmass))
    #     if self.merging_want_primordial == True:
    #         self.sol_rates_early_binaries = self.rates_primordial()
    #         print("Merging rates of primordial binaries have been calculated")
    #         print("====")

    #     if self.merging_want_clusters == True:
    #         self.sol_rates_cluster = self.rates_clusters()
    #         print("Merging rates of binaries formed by capture in clusters have been calculated")
    #         print("====")

    #     print("End of code, at the moment...  Thank you for having used PrimBholes")




if __name__ == "__main__":

    from power_spectrum import PowerSpectrum
    import matplotlib.pyplot as plt
    from matplotlib import ticker, cm
    
    masses =  10**np.linspace(-10,0, 100)  

    # PS_model = PowerSpectrum.gaussian(kp=2.e6, As=0.0205, sigma=1.)
    # PS_func =  PS_model.PS
    
    # def PS_func(kk):
    #     AsPBH, kp, sigma = [0.01, 1.e7, 0.25]
    #     # AsPBH *= 1.183767
    #     return AsPBH * np.exp(- np.log(kk / kp) ** 2 / (2 * sigma ** 2))
    

    # Model A: Gaussian
    sig =  0.3
    As = 0.01*sig
    kp = 5e9
    PS_model = PowerSpectrum.gaussian(As=As, sigma=sig, kp=kp)
    PS_func =  PS_model.PS_plus_vacuum        # This is the default to calculate sigma and fPBH

    ks = 10**np.linspace(6,12, 300)
    fig, axs = plt.subplots(1,3, figsize=(15,5)) 
    ax = axs[0]
    ax.plot(ks , PS_func(ks))
    # ax.set_ylim(1e-10, 2)
    ax.set_ylabel("Power spectrum")
    ax.set_xlabel("k")
    ax.set_xscale("log")
    ax.set_yscale("log")

    my_abundances = CLASSabundances(ps_function=PS_func)
    fpbhs = my_abundances.get_fPBH(masses)
    fpbh_integrated =  1 # my_abundances.get_integrated_fPBH()


    print(f"integrated fpbh = {fpbh_integrated}")


    if fpbh_integrated > 10 : 
        print(f'\n\n fpbh_integrated is way too large,  {fpbh_integrated}\n')
        raise

    if np.any(fpbhs) > 10 : 
        print('\n\n fpbhs is way too large.\n')
        raise




   # EARLY BINARIES

    sol = MergerRates().get_rates_early_binaries(fpbh_integrated, masses, fpbhs) 
    
    ax = axs[1]
    Z =  np.log10(sol).T
    floor = 0.0
    Z[(Z<floor)] = floor
    cs=ax.contourf(np.log10(masses),np.log10(masses),Z, levels=10) 
    ax.set_xlabel(r'$\log_{10} \, m_1 /M_\odot $')
    ax.set_ylabel(r'$\log_{10} \, m_2 /M_\odot $')
    ax.set_title("Merging rates for early binaries")
    cbar = fig.colorbar(cs)
    cbar.set_label(r'log$_{10}$ Rates  [$yr^{-1}Gpc^{-3}$]', rotation=90)
    ax.grid(True)


    # LATE BINARIES

    sol = MergerRates().get_rates_late_binaries(fpbh_integrated, masses, fpbhs) 

    ax = axs[2]
    Z =  np.log10(sol).T
    floor = 0.0
    Z[(Z<floor)] = floor
    cs=ax.contourf(np.log10(masses),np.log10(masses),Z, levels=10) 
    ax.set_xlabel(r'$\log_{10} \, m_1 /M_\odot $')
    ax.set_ylabel(r'$\log_{10} \, m_2 /M_\odot $')
    ax.set_title("Merging rates for late binaries")
    cbar = fig.colorbar(cs)
    cbar.set_label(r'log$_{10}$ Rates  [$yr^{-1}Gpc^{-3}$]', rotation=90)
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()