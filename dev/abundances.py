import numpy as np
import scipy.constants as const
import scipy.special as special
from scipy.special import erfc
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.integrate import dblquad
import scipy.optimize as opt


import sys, os
FILEPATH = os.path.realpath(__file__)[:-21]
sys.path.append(FILEPATH + "/src")
sys.path.append("./src")
# print(f"FILEPATH = {FILEPATH}")


from user_params import cosmo_params, physics_units, PBHForm

from power_spectrum import PowerSpectrum

from threshold import ClassPBHFormationMusco20



import matplotlib.pyplot as plt

class CLASSabundances:
    def __init__(self, 
                    powerspectrum=None, 
                    PS_function = None,
                    scaling=False,
                    gaussian=True):
        
        self.Pkm = powerspectrum if powerspectrum else PowerSpectrum.gaussian() 
        self.Pk = PS_function if PS_function else self.Pkm.PS_plus_vaccumm
          
        # self.Pk = powerspectrum if powerspectrum else PowerSpectrum.default().PS
        self.Pkrescaling = scaling
        self.Pkscalingfactor = scaling if isinstance(scaling, float) else 1.0 
        self.Gaussian = gaussian
        self.rescaling_is_done = True if isinstance(scaling, float) else False
        # self.rm =  rm if rm else PowerSpectrum.default


    def myPk(self, k):

        # return self.Pk(k * self.Pkm.kp)
        return self.Pk(k)

    
    def get_beta(self, mPBH):

        # This assumes Gaussian PDF for the overdensity-powerspectra

        # dcrit = self.get_dcrit()        
        # sigma = self.get_variance(mPBH) **0.5
        # nu = dcrit/sigma

        dcrit = self.get_dcrit()  # 0.8
        # sigma = 0.0905
        sigma = self.get_variance(mPBH) **0.5

        # print(f" USED mPBH = {mPBH},  sigma = {sigma}  ")

        # nu = dcrit/sigma     
        # beta = np.exp(-0.5*nu**2)/np.sqrt(2*np.pi*nu**2)   #approx
        # beta0 = np.copy(beta)   


        if isinstance(sigma, (float, int)) : 
            sigma = np.array([sigma])

        betas = []
        for sig in sigma: 

        
            def _integrator_PDF(delta):
                # returns the dark matter density fraction of PBH today f(m_PBH)
                return  1/np.sqrt(2*np.pi*sig**2) * np.exp(-0.5*(delta/sig)**2)


            # Integrate
            init = 0 # -np.infty
            end = 2
            sol_D = integrate.quad(_integrator_PDF, init, end,  limit=100000, limlst=10000)[0]
            sol_U = integrate.quad(_integrator_PDF, dcrit, end,  limit=100000, limlst=10000)[0]
            b_altern = np.exp(-0.5*(dcrit/sig)**2)/np.sqrt(2*np.pi*(dcrit/sig)**2)
            beta = sol_U/sol_D if np.abs(sol_D) > 0 else  b_altern     #erfc(dcrit/np.sqrt(2*sig**2))

            print("altern ", b_altern)


            betas.append(beta)

        betas = np.array(betas)
        return betas

    def get_overdensity_powerspectrum(self, k, mPBH):

        # This assumes standard thermal history in a Radiation Domination

        k_PBH = self.get_kPBH(mPBH)
        delta_PS = (16./81) * (k/k_PBH)**4 * self.myPk(k)

        # print(k, delta_PS)
        return delta_PS

    
    def get_window_function(self, k, r):

        # NOTE:  The effect on the choice of the window function (plus transfer function) is enourmous

        W = np.exp( -0.5 * (k * r)**2 )        # Assuming Gaussian Window function 

        sq = 1.
        arg = k * r * sq
        W2 =  3 * (np.sin(arg) - arg * np.cos(arg)) / arg ** 3

        sq = 1. / np.sqrt(3)
        arg = k * r * sq
        T2 = 3 * (np.sin(arg) - arg * np.cos(arg)) / arg ** 3


        # return  W2 * T2      
        return W



    def get_variance(self, mPBH):

        # def _integrator_variance(k):
        #     return self.get_window_function(k, mPBH)**2 * self.get_overdensity_powerspectrum(k, mPBH) / k

        if isinstance(mPBH, (float, int)) : 
            mPBH = np.array([mPBH])

        vs = []
        for Mass in mPBH: 

        
            rPBH = 1/ self.get_kPBH(Mass) 

            # rPBH =     Mass*16*physics_units.G/0.2
            rPBH = 2*Mass*physics_units.G/(physics_units.c)**2    / physics_units.mpc

            # mH = Mass / PBHForm.ratio_mPBH_over_mH
            # kk = PBHForm.kmsun / mH ** (0.5)  # S. Clesse: to be checked
            # rPBH = 1/kk

            # rPBH =  Mass/physics_units.m_sun   *  PBHForm.kmsun / 1000 * physics_units.mpc
            # rPBH = 1/(2.e6)

           

            print(f"R_PBH  = {rPBH}" )

            def _integrator_variance(k):
                
                # factor = 2.*np.pi**2 /(k**3)
                kp = 1 # 2.e6
                kk = k / kp

                return   (16./81) * (kk*rPBH)**4 * self.get_window_function(kk, rPBH)**2 * self.myPk(kk * kp )  /kk   

          


            # Integrate
            kmin = 0  # -np.infty
            kmax = np.infty
            sol = integrate.quad(_integrator_variance, kmin, kmax,  limit=100000, limlst=10000)
            # sol = integrate.quad(_integrator_variance, kmin, kmax)
            variance =   sol[0]

            # variance = 0.0046    # (sweet number)

            print(f" USED mPBH = {Mass},  SOL variance = {variance}  ")
            vs.append(variance)

            # # verbose: 
            x = 10**np.linspace(-6,18,200)
            plt.plot(x, _integrator_variance(x))
            plt.xscale("log")
            plt.yscale("log")
        
        plt.show()
        
        vs = np.array(vs)
        print(f"we found a variance of  {vs}")

        return vs
    
    def get_kPBH(self, mPBH):

        # TODO: set params in default params:  Choice method

        gamma = 0.2
        g_star = 106.75
        Msun = 1.989e30
        factor = 2.4*1e5  #/ physics_units.mpc      # Franciolini thesis



        kPBH = factor * (gamma/0.2)**0.5  * (g_star/106.75)**(-1./12)  * (mPBH / 30/Msun)**(-0.5)     ## Check exponent -1/12, why not -1/4 ?

        # print("F:", kPBH, mPBH, factor)

        # mH = mPBH / PBHForm.ratio_mPBH_over_mH
        # RH =  2 * mH * physics_units.G / (physics_units.c)**2   #/ PBHForm.kmsun      ### PBHForm.kmsun * (1/mH)

        # kPBH = 1/RH

        # print("S" , kPBH, RH,  mPBH)

        return  kPBH  

    def get_dcrit(self):

        # TODO: implement or call threshold class

        dcrit_default =  0.41  # 0.8

        return dcrit_default


    def get_fPBH(self, mPBH):

        gamma = 0.2
        g_star = 106.75
        Msun = 1.989e30
        factor = 1.65*1e8 
        h = 0.68

        fPBH = factor * (gamma/0.2)**0.5  * (g_star/106.75)**(-1./4)  * (h/0.68)**(-2) * (mPBH / Msun)**(-0.5) * self.get_beta(mPBH)

        return fPBH




    # def get_rm(self):

    #     t = 0.1
    #     guess= 1.0

    #     def _TransferFunction(k):
    #         sq = 1. / np.sqrt(3)
    #         arg = k * t * sq
    #         return 3 * (np.sin(arg) - arg * np.cos(arg)) / arg ** 3

    #     def _P(k):
    #         # PkS = ClassPkSpectrum(pkmodel=self.Pk_model, cm=self.cp)

    #         # print("!",  self.Pk_func[0], " and ",  self.Pkscalingfactor)
    #         Pk = self.Pk   
    #         PP = Pk(k)     #* self.Pkscalingfactor
    #         T =  _TransferFunction(k)
    #         return 2.*np.pi**2 /(k**3) * PP * T**2
        
    #     def func(rm):
    #         integrand = lambda k: k ** 2 * (
    #                     (k ** 2 * rm ** 2 - 1) * np.sin(k * rm) / (k * rm) + np.cos(k * rm)) * _P(k)
    #         integ = integrate.quad(integrand, 0, np.inf, limit=100000, limlst=10000)
    #         return integ[0]

        
    #     sol = opt.root(func, x0=guess)
    #     root = sol.x
    #     success = sol.success
    #     if success:
    #         pass
    #     else:
    #         root, success = opt.bisect(func, 0., 1e8, rtol=1.e-5, maxiter=100, full_output = True)
    #         # success = True
        
    #     return root




class __deprecated_CLASSabundances:
    def __init__(self, 
                    powerspectrum=None, 
                    scaling=False,
                    gaussian=True):
        self.Pk = powerspectrum if powerspectrum else PowerSpectrum.default
        self.Pkrescaling = scaling
        self.Pkscalingfactor = scaling if isinstance(scaling, float) else 1.0 
        self.Gaussian = gaussian
        self.rescaling_is_done = True if isinstance(scaling, float) else False



    def get_logbeta(self, mPBH, use_thermal_history='default'):  #TODO

        if use_thermal_history == 'default':
            use_thermal_history = self.use_thermal_history

        if self.Pkrescaling == True and self.rescaling_is_done == False:
            self.calc_scaling(mPBH)

        PkS = self.Pk

        # returns the density fraction of PBH at formation \beta(m_PBH)
        mH = mPBH / self.ratio_mPBH_over_mH
        kk = self.kmsun / mH ** (0.5)  # S. Clesse: to be checked
        limit_for_using_erfc = 20.                    #TODO hardcoded
        Pofk = PkS.Pk(kk) * self.Pkscalingfactor

        # if verbose: print("323 :: ", Pofk, kk)

        zetacr = self.get_zetacr(mPBH,  use_thermal_history=use_thermal_history)

        if (self.Gaussian):
            argerfc = zetacr / (2. * Pofk) ** (1. / 2.)
            logbeta = np.zeros_like(argerfc)
            mask = (argerfc < limit_for_using_erfc)
            logbeta[mask] = np.log10(erfc(argerfc[mask]))
            logbeta[~mask] = -np.log10(argerfc[~mask] * np.sqrt(np.pi)) - argerfc[~mask] ** 2 / np.log(10.)


        else:
            raise (ValueError, "Non-Gaussian effects are not implemented yet")

        return logbeta

    def get_k(self, mPBH):
        # Returns the scale (in [Mpc^-1] )  for a given PBH mass
        mH = mPBH / self.ratio_mPBH_over_mH
        kk = self.kmsun * np.sqrt(1. / mH)  # S. Clesse: to check
        return kk

    def calc_scaling(self, mPBH):
        # Compute the Rescaling of the power spectrum to get the forcefPBH

        
        def _rootfunction_logfPBH(scaling):
            self.Pkscalingfactor = 10. ** scaling
            function = self.get_logfPBH(mPBH) - np.log10(self.forcedfPBH)
            return function

        if self.Pkrescaling == False:
            raise Exception('Error: rescaling of the power spectrum when not allowed (ie. Pkrescaling = False)')
        self.rescaling_is_done = True  # Needed to stop/avoid bucle

        if verbose > 1:
            print(":: Rescaling of the power spectrum to get f_PBH =", self.forcedfPBH)

        sol = opt.bisect(_rootfunction_logfPBH, -1., 1., rtol=1.e-5, maxiter=100)
        self.Pkscalingfactor = 10. ** sol

        if verbose > 1:
            print(":: After rescaling, I get a total abundance of PBHs: fPBH=", self.get_fPBH(mPBH))
            print(":: Rescaling factor=", self.Pkscalingfactor)
            print(":: zeta_crit (radiation) = ", self.get_deltacr())
            print("====")

        return self.Pkscalingfactor


    def get_nonrescaled_logfofmPBH(self, mPBH, mHeq = 2.8e17):   #TODO   mHeq is hardcoded
        # returns log_10 of the dark matter density fraction of PBH today f(m_PBH)
        cp = self.cp
        logbeta = self.get_logbeta(mPBH)
        logfofmPBH = logbeta + np.log10(
            (mHeq / (mPBH / self.ratio_mPBH_over_mH)) ** (1. / 2.) * 2. / (cp.Omc / (cp.Omc + cp.Omb)))
        return logfofmPBH

    def get_logfofmPBH(self, mPBH, mHeq = 2.8e17):   #TODO   mHeq is hardcoded
        # returns log_10 of the dark matter density fraction of PBH today f(m_PBH)
        if self.Pkrescaling == True and self.rescaling_is_done == False:
            self.calc_scaling(mPBH)
        logfofmPBH = self.get_nonrescaled_logfofmPBH(mPBH, mHeq)
        return logfofmPBH

    def get_fofmPBH(self, mPBH):
        return 10**self.get_logfofmPBH(mPBH)

    def get_fPBH(self, mPBH):

        def _integrator_foflogmPBH(logmPBH):
            # returns the dark matter density fraction of PBH today f(m_PBH)
            logfofmPBH = logfofmPBH_interp(logmPBH)
            foflogmPBH = 10. ** logfofmPBH
            return foflogmPBH

        logmPBHtable = np.log10(mPBH)
        logmass_min = np.min(logmPBHtable)
        logmass_max = np.max(logmPBHtable)
        logfofmPBHtable = self.get_nonrescaled_logfofmPBH(mPBH)

        # Integrate
        logfofmPBH_interp = interp1d(logmPBHtable, logfofmPBHtable)
        fPBHsol = integrate.quad(_integrator_foflogmPBH, logmass_min, logmass_max, epsrel=0.001)
        fPBH = fPBHsol[0]

        return fPBH

    def get_logfPBH(self, mPBH_table):
        logfPBH = np.log10(self.get_fPBH(mPBH_table))
        return logfPBH

    def get_Pk(self, k):
        return self.Pk(k)



#######################################

if __name__ == "__main__":
    test = 0
    Msun = physics_units.m_sun

    PS_model = PowerSpectrum.gaussian() 
    # PS_model = PowerSpectrum.powerlaw() 

    a = CLASSabundances(powerspectrum=PS_model, PS_function = PS_model.PS_plus_vaccumm)

    mass = np.array([0.01, 1, 10, 1e3])  * Msun
    # mass = 30 * Msun

    beta = a.get_beta(mass)
    fpbh = a.get_fPBH(mass)

    print(f" default beta = {beta},  and f_PBH = {fpbh}")

