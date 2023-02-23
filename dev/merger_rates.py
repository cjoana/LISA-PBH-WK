"""
Code to evaluate merger rates given a mass distribution f(m1, m2, z), with PBH masses m1, m2 at a redshift z. 
Basic models are included and others can be expanded easily. 
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
FILEPATH = os.path.realpath(__file__)[:-20]
sys.path.append(FILEPATH + "/src")
sys.path.append("./src")
print(f"FILEPATH = {FILEPATH}")


# import abundances
from abundances import CLASSabundances
from user_params import cosmo_params, physics_units


class MergerRates(CLASSabundances): 

    def __init__(self, **kargs): 
        super().__init__(**kargs)

        self.Rclust = 400               # TODO : hardcode
        self.fsup = 0.0025              # TODO : hardcode
        self.ratio_mPBH_over_mH = 0.8   # TODO : hardcode

        self.Omc = cosmo_params.Omc
        self.Omb = cosmo_params.Omb


        # output
        self.sol_rates_primordial = None
        self.sol_rates_cluster = None        



    def rates_primordial_binary(self, m1, m2):

        rates = norm * self.fsup * self.get_fPBH(m1) * self.get_fPBH(m2)  * \
                    (m1 + m2) ** (-32. / 37.) * (m1 * m2 / (m1 + m2) ** 2) ** (-34. / 37.)
        return rates

        #### 
    def get_rates_primordial(self, masses):
        # Computes the merging rates of primordial binaries
        norm = 1.6e6
        Nmass = len(masses)
        rates = np.zeros([Nmass, Nmass])
        for ii in range(Nmass):
            m1 = masses[ii]
            for jj in range(ii):
                m2 = masses[jj]
                rates[ii,jj] = self.rates_primordial_binary(m1, m2)
        self.sol_rates_primordial = rates
        return rates


    def rates_cluster_binary(self, m1, m2):
        norm = self.Rclust
        rates = norm * self.fsup * self.get_fPBH(m1) * self.get_fPBH(m2)  * \
                    (m1 + m2) ** (-32. / 37.) * (m1 * m2 / (m1 + m2) ** 2) ** (-34. / 37.)
        return rates

    def get_rates_clusters(self, masses):
        # Computes the merging rates for tidal capture in PBH clusters
        Nmass = len(masses) 
        rates = np.zeros((Nmass, Nmass))
        for ii in range(Nmass):
            m1 = masses[ii]
            for jj in range(ii):
                m2 = masses[jj]
                rates[ii,jj] = self.rates_cluster_binary(m1, m2)
        self.sol_rates_cluster = rates
        return rates

    def eval_oldcode(self):
        # Compute the merging rates
        print("Step 3:  Computation of the PBH merging rates")
        print("====")
        self.sol_rates_primordial = np.zeros((self.Nmass, self.Nmass))
        self.sol_rates_cluster = np.zeros((self.Nmass, self.Nmass))
        if self.merging_want_primordial == True:
            self.sol_rates_primordial = self.rates_primordial()
            print("Merging rates of primordial binaries have been calculated")
            print("====")

        if self.merging_want_clusters == True:
            self.sol_rates_cluster = self.rates_clusters()
            print("Merging rates of binaries formed by capture in clusters have been calculated")
            print("====")

        print("End of code, at the moment...  Thank you for having used PrimBholes")




if __name__ == "__main__":

    from power_spectrum import PowerSpectrum
    import matplotlib.pyplot as plt
    from matplotlib import ticker, cm
    
    masses =  10**np.linspace(-3,4, 50)  

    # PS_model = PowerSpectrum.gaussian(kp=2.e6, As=0.0205, sigma=1.)
    # PS_func =  PS_model.PS
    
    def PS_func(kk):
        AsPBH, kp, sigma = [0.0205, 2.e6, 1.]
        # AsPBH *= 1.183767
        return AsPBH * np.exp(- np.log(kk / kp) ** 2 / (2 * sigma ** 2))
    

    ks = 10**np.linspace(2,14, 100) 
    plt.plot(ks , PS_func(ks))
    plt.ylim(1e-10, 2)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()


    sol = MergerRates(PS_function=PS_func).get_rates_clusters(masses)
    
    
    figRprim = plt.figure()
    # figRprim.patch.set_facecolor('white')
    ax = figRprim.add_subplot(111)
    Z =  np.transpose(np.log10(sol))
    floor = 1e-2
    Z[Z<floor] = np.nan
    cs=ax.contourf(np.log10(masses),np.log10(masses),Z, levels=10) 
    plt.title("Merging rates for primordial binaries")
    cbar = figRprim.colorbar(cs)
    cbar.set_label(r'$yr^{-1}Gpc^{-3}$', rotation=90)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    #plt.ylim(1.e-4,1.e1)
    plt.xlabel(r'$\log \, m_1 /M_\odot $')
    plt.ylabel(r'$\log \, m_2 /M_\odot $')
    plt.grid(True)
    # figRprim.savefig(figdir + '/Rprim.png',facecolor=figRprim.get_facecolor(), edgecolor='none',dpi=600)
    plt.show()
