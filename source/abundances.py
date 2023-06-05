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
FILEPATH = os.path.realpath(__file__)[:-18]
sys.path.append(FILEPATH + "/src")
sys.path.append("./src")
print(f"FILEPATH = {FILEPATH}")


from user_params import cosmo_params, physics_units, PBHForm

from power_spectrum import PowerSpectrum
from threshold import ClassThresholds

# from threshold import ClassPBHFormationMusco20




class CLASSabundances:
    def __init__(self, 
                    powerspectrum=None, 
                    PS_function = None,
                    scaling=False,
                    gaussian=True,
                    threshold_method='standard',
                    thermal_history = True):
        
        self.PS_model = powerspectrum if powerspectrum else PowerSpectrum.gaussian() 
        self.PS_func = PS_function if PS_function else self.PS_model.PS_plus_vaccumm          #TODO warning: perhaps is better set somewhere else. 
          
        self.PS_rescaling = scaling
        self.PS_scalingfactor = scaling if isinstance(scaling, float) else 1.0 
        self.Gaussian = gaussian
        self.rescaling_is_done = True if isinstance(scaling, float) else False
        self.threshold_method=threshold_method
        self.thermal_history = thermal_history


    def myPk(self, k):
        return self.PS_func(k)

    
    def get_beta(self, mPBH, method="integration"):

        if method == "semianalytical":
            return self.get_beta_anal_approx(mPBH)        #Testing 

        if method == "integration":
            dcrit = self.get_dcrit(mPBH=mPBH)
            sigma = self.get_variance(mPBH) **0.5       

            if isinstance(sigma, (float, int)) : 
                sigma = np.array([sigma])
            if isinstance(dcrit, (float, int)) :
                dcrit = dcrit * np.ones_like(sigma)

            betas = []
            for i_s, sig in enumerate(sigma): 

                do_integration = False    #TODO : specify! (I checked, gives aprox the same)
                if do_integration: 
                    def _integrator_PDF(delta):
                        # returns the dark matter density fraction of PBH today f(m_PBH)
                        return  1/np.sqrt(2*np.pi*sig**2) * np.exp(-0.5*(delta/sig)**2)

                    # Integrate
                    dc = dcrit[i_s]
                    init = 1e-8
                    end = np.infty
                    sol_D = integrate.quad(_integrator_PDF, init, end,  limit=100000, limlst=10000)[0]
                    sol_U = integrate.quad(_integrator_PDF, dc, end,  limit=100000, limlst=10000)[0]
                    b_altern = np.exp(-0.5*(dc/sig)**2)/np.sqrt(2*np.pi*(dc/sig)**2)
                    beta = sol_U/sol_D if np.abs(sol_D) > 0 else  b_altern     #erfc(dcrit/np.sqrt(2*sig**2))
                else:
                    beta = np.exp(-0.5*(dcrit[i_s]/sig)**2)/np.sqrt(2*np.pi*(dcrit[i_s]/sig)**2)   #TODO:  SPi paper or Sebs:  Check if 2gamma is needed

                betas.append(beta)

            betas = np.array(betas)


            return betas
        else:
            m = f"error in get_beta: method {method} is not configured." 
            raise Exception(m)


    def get_beta_anal_approx(self, mPBH):

        #params to set
        ratio_mPBH_over_mH = 0.8
        kmsun = 2.1e6
        limit_for_using_erfc =  20. 

        dcrit = self.get_dcrit(mPBH=mPBH)
        mH = mPBH / ratio_mPBH_over_mH
        kk = kmsun / mH ** (0.5)  # S. Clesse: to be checked
        Pofk = self.PS_func(kk)
        
        
        argerfc = dcrit / (2. * Pofk) ** (1. / 2.)
        logbeta = np.zeros_like(argerfc)
        mask = (argerfc < limit_for_using_erfc)
        logbeta[mask] = np.log10(erfc(argerfc[mask]))
        logbeta[~mask] = -np.log10(argerfc[~mask] * np.sqrt(np.pi)) - argerfc[~mask] ** 2 / np.log(10.)

        beta = 10**logbeta
        return beta


    def get_overdensity_powerspectrum(self, k, mPBH):

        # This assumes standard thermal history in a Radiation Domination

        k_PBH = self.get_kPBH(mPBH)
        delta_PS = (16./81) * (k/k_PBH)**4 * self.myPk(k)

        print("WARNING", k, delta_PS)
        return delta_PS


    def get_window_function(self, k, r, method="default"):

        # NOTE:  The effect on the choice of the window function (plus transfer function) is enourmous

        if method=="gaussian":
            W = np.exp( -0.5 * (k * r)**2 )       
            return W

        if method=="default":
            sq = 1.
            arg = k * r * sq
            Wtilde =  3 * (np.sin(arg) - arg * np.cos(arg)) / arg ** 3

            sq = 1. / np.sqrt(3)
            arg = k * r * sq
            Ttilde = 3 * (np.sin(arg) - arg * np.cos(arg)) / arg ** 3


            return  Wtilde * Ttilde      
        
        else:
            m = f"error in get_window_function: method {method} is not configured." 
            raise Exception(m)
        

    def get_scalesize(self, mPBH):

        #TODO: put params  outside 
        ratio_mPBH_over_mH = 0.8
        kmsun = 2.1e6
        mH = mPBH / ratio_mPBH_over_mH

        k_scale = kmsun / mH**(0.5)  
        r_pbh = 1./k_scale

        return r_pbh


    def get_variance(self, mPBH):


        if isinstance(mPBH, (float, int)) : 
            mPBH = np.array([mPBH])

        vs = []
        for Mass in mPBH: 

            RH = self.get_scalesize(Mass)

            def _integrator_variance(k):
                out =  self.get_window_function(k, RH)**2 * self.myPk(k) * (k*RH)**4   /k                 
                return   out  

            kint = 1/RH
            # Integrate
            kmin = kint * 1e-3    # Clever integration countours ( + speeds & precition ) making use of window-function effect
            kmax = kint * 1e3
            sol, err = integrate.quad(_integrator_variance, kmin, kmax,  epsabs= 1e-25, epsrel= 1e-5, limit=int(1e4), limlst=int(1e2) )
            variance =   (16./81) *  sol
            vs.append(variance)

        vs = np.array(vs)

        return vs
    

    def get_dcrit(self, mPBH = False):

        # TODO: implement or call threshold class
        # self.threshold_method = "Musco20"
        # self.threshold_method = "standard"
        Msun = physics_units.m_sun

        if self.threshold_method == "standard":
            if self.thermal_history: return ClassThresholds.standard(PS_func=self.PS_func).get_deltacr_with_thermalhistory(mPBH)
            else: return ClassThresholds.standard(PS_func=self.PS_func).get_deltacr()

        elif self.threshold_method == "Musco20":
            if self.thermal_history: return ClassThresholds.Musco20(PS_func=self.PS_func).get_deltacr_with_thermalhistory(mPBH)
            else: return ClassThresholds.Musco20(PS_func=self.PS_func).get_deltacr()

        else:
            raise ("selected method for evaluating PBH threshold is not yet suported.")
            dcrit_default =  0.41   # In Pi-Wang 2022 they use dc = 0.41 / Similar to Harada or Escriva
            # dcrit_default =  1.02   # from Clesse default zetacr
            return dcrit_default


    def get_fPBH(self, mPBH):

        # TODO: set in params
        gamma = 0.2
        g_star = 106.75
        Msun = 1.989e30
        factor = 1.65*1e8 
        h = 0.68

        fPBH = factor * (gamma/0.2)**0.5  * (g_star/106.75)**(-1./4) * \
                        (h/0.68)**(-2) * (mPBH / Msun)**(-0.5) * self.get_beta(mPBH)

        return fPBH



#######################################

if __name__ == "__main__":


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



    def k_of_m(mass):
        ratio_mPBH_over_mH = 0.8
        kmsun = 2.1e6
        mH = mass / ratio_mPBH_over_mH
        kk = kmsun / mH ** (0.5)  # S. Clesse: to be checked
        return kk
    def m_of_k(k):

        ratio_mPBH_over_mH = 0.8
        kmsun = 2.1e6

        mass = (kmsun/k)**2 *  ratio_mPBH_over_mH
        return mass




    test = 0
    Msun = physics_units.m_sun

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
    
    ## Select threshold calc method
    a = CLASSabundances(PS_function = PS_func, threshold_method="standard")
    # a = CLASSabundances(PS_function = PS_func, threshold_method="Musco20")

    ## Params range: 
    # mass = 10**np.linspace(-10,20, 1000)  #* Msun
    mass = 10**np.linspace(-6,8, 500)  #* Msun

    floor = 1e-8
    beta = a.get_beta(mass)  #+ floor
    fpbh = a.get_fPBH(mass)  + floor
    sigma = a.get_variance(mass)


    fig, axs = plt.subplots(4,1, figsize=(8,8))

    kk =  k_of_m(mass)
    ax = axs[0]
    ax.plot(kk, PS_func(kk))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$P_\zeta(k)$")
    ax.set_ylim(1e-9, 1.5)
    ax.axhline(1, color="k", ls="--", alpha=0.5)
    

    ax = axs[1]
    ax.plot(kk, sigma)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\sigma$")
    ax.set_xlabel(r"$k\ [Mpc^{-1}]$")

    ax = axs[2]
    ax.plot(mass, beta)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\beta$")
    ax.set_xlabel(r"$m_{PBH}\ [?]$")
    ax.set_xlim(max(mass), min(mass))
    ax.set_ylim(beta.max()*1e-8, beta.max()*10)

    ax = axs[3]
    ax.plot(mass, fpbh)
    ax.axhline(1, color="k", ls="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$f_{pbh}$")
    ax.set_xlabel(r"$m_{PBH}\ [?]$")
    ax.set_xlim(max(mass), min(mass))

    plt.tight_layout()
    plt.show()

