"""
Second order SGWB from inflationary primordial scalar powerspectrum
"""
import numpy as np
from scipy.integrate import dblquad


import sys, os
FILEPATH = os.path.realpath(__file__)[:-24]
sys.path.append(FILEPATH + "/src")
sys.path.append("./src")
print(f"FILEPATH = {FILEPATH}")


from user_params import cosmo_params, physics_units

class SecondOrderSGWB():

    def __init__(self, PS_func, PS_scalingfactor=1):        
        # power spectra params
        self.PS_func = PS_func
        self.PS_scalingfactor = PS_scalingfactor

        #physics constants 
        self.c =physics_units.c
        self.mpc =physics_units.mpc
        #cosmo params
        self.kp = cosmo_params.kp
        self.ratio_mPBH_over_mH = 0.8   # TODO read from PBHform params BUT:  WHY Need this here?!!

        # output
        self.freq_2ndOmGW = None
        self.OmGW_2ndOmGW = None


    # Functions to compute the SGWB from second order perturbations
    def IC2(self, d, s):
        return -36 * np.pi * (s**2 + d**2 - 2)**2 / (s**2 - d**2)**3 * np.heaviside(s - 1, 1)

    def IS2(self, d, s):
        return -36 * (s**2 + d**2 - 2) / (s**2 - d**2)**2 * \
            ( (s**2 + d**2 - 2) / (s**2 - d**2) * np.log((1 - d**2) / np.absolute(s**2 - 1)) + 2)

    def IcsEnvXY(self, x, y):
        return np.sqrt(self.IC2(np.absolute(x - y) / (3**0.5), np.absolute(x + y) / (3**0.5))**2 + \
                self.IS2(np.absolute(x - y) / (3**0.5), np.absolute(x + y) / (3**0.5))**2)

    # Integral returning the spectrum
    def compint(self, kvval, sigmaps):

        kcoef = self.mpc / self.c

        value, error = dblquad(lambda x, y:
                                x**2 / y**2 * (1 - (1 + x**2 - y**2)**2 / (4 * x**2))**2
                                * self.PS_func(kvval * kcoef) * self.PS_scalingfactor  # PS(kvval*x)
                                * self.PS_func(kvval * kcoef) * self.PS_scalingfactor  # PS(kvval*y)
                                * self.IcsEnvXY(x, y)**2
                                ,
                                10**(- 4 * sigmaps) / kvval, 10**(4 * sigmaps) / kvval, lambda x: np.absolute(1 - x)
                                ,
                                lambda x: 1 + x)
        return value

    def get_OmegaGWs(self, kvals, recalculate=1):
    
        if (not recalculate) and (not isinstance(self.OmGW_2ndOmGW, bool)):
            return self.OmGW_2ndOmGW

        ks = self.kp / self.mpc * self.c
        sigmaps = 0.5

        kres = np.array([self.compint(kk, sigmaps) for kk in kvals])
        # TODO: 
        # coefficient due to thermal history see Eq. (2.11) https://arxiv.org/pdf/1810.12224.pdf
        # to be updated depending on the reference peak of the spectrum, to integrated with the rest of the code
        Omega_r_0 = 2.473 * 1e-5
        norm = self.ratio_mPBH_over_mH * Omega_r_0 / 972.
        self.OmGW_2ndOmGW = norm * kres

        return self.OmGW_2ndOmGW

    def get_frequencies(self, kvals):

        ks = self.kp / self.mpc * self.c
        self.freq_2ndOmGW = ks * kvals / 2. /np.pi

        return self.freq_2ndOmGW

    
    def eval_oldcode(self, kvals):

        # Compute the SGWB from 2nd order perturbations
        print("Step 2:  Computation of the GW spectrum from denstiy perurbations at second order")
        print("Can take several minutes depending on the value of nk...")
        print("====")

        ks = self.kp / self.mpc * self.c
        sigmaps = 0.5

        kres = np.array([self.compint(kk, sigmaps) for kk in kvals])
        # TODO: 
        # coefficient due to thermal history see Eq. (2.11) https://arxiv.org/pdf/1810.12224.pdf
        # to be updated depending on the reference peak of the spectrum, to integrated with the rest of the code
        Omega_r_0 = 2.473 * 1e-5
        norm = self.ratio_mPBH_over_mH * Omega_r_0 / 972.

        #  self.k_2ndOmGW= ks*kvals
        self.freq_2ndOmGW = ks * kvals / 2. /np.pi
        self.OmGW_2ndOmGW = norm * kres







#######################################

if __name__ == "__main__":

    print(" Hello, this is the class to calculate 2nd Order GWs (to SGWB), induced from the scalar powerspectra.")

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    from power_spectrum import PowerSpectrum


    #Specify the plot style
    mpl.rcParams.update({'font.size': 10,'font.family':'serif'})
    mpl.rcParams['xtick.major.size'] = 7
    mpl.rcParams['xtick.major.width'] = 1
    mpl.rcParams['xtick.minor.size'] = 3
    mpl.rcParams['xtick.minor.width'] = 1
    mpl.rcParams['ytick.major.size'] = 7
    mpl.rcParams['ytick.major.width'] = 1
    mpl.rcParams['ytick.minor.size'] = 3
    mpl.rcParams['ytick.minor.width'] = 1
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['ytick.right'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rc('text', usetex=True)

    mpl.rcParams['legend.edgecolor'] = 'inherit'


    ## Model A: Gaussian
    sig =  0.25
    As = 0.01*sig
    kp = 1e7
    PS_model = PowerSpectrum.gaussian(As=As, sigma=sig, kp=kp)
    
    ## Model B : axion_gauge
    # PS_model = PowerSpectrum.axion_gauge()    
    # PS_model = PowerSpectrum.axion_gauge(As=As, sigma=sig, kp=kp)
    
    ## Select with vacuum
    PS_func =  PS_model.PS_plus_vaccumm        # This is the default to calculate sigma and fPBH
    
    # Call the class 
    SOGW = SecondOrderSGWB(PS_func=PS_func)

    kvals = 10**np.linspace(-6,15, 1000)  


    fig, axs = plt.subplots(1,2, figsize=(8,5))
    ###################################################

    ax = axs[0]
    x = kvals
    y = PS_func(kvals)
    ###################
    ax.plot(x, y)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$P_\zeta(k)$")
    ax.set_xlabel(r"$k\ [Mpc^{-1}]$")
    # ax.set_ylim(1e-9, 1.5)
    ax.axhline(1, color="k", ls="--", alpha=0.5)
    ###################################################

    ax = axs[1]
    x = SOGW.get_frequencies(kvals)
    y = SOGW.get_OmegaGWs(kvals)
    ###################
    ax.plot(x, y)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\Omega_{\rm GW}$")
    ax.set_xlabel(r"freq $\ [Hz]$")
    # ax.set_ylim(1e-9, 1.5)
    # ax.axhline(1, color="k", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

