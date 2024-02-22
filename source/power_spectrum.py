import numpy as np
import scipy.constants as const
import scipy.special as special
from scipy.special import erfc
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.integrate import dblquad
import scipy.optimize as opt
# from munch import Munch


import sys, os
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
SOURCEPATH = os.path.abspath(os.path.join(ROOTPATH, 'source'))
PARAMSPATH = os.path.abspath(os.path.join(ROOTPATH, 'params'))
PLOTSPATH = os.path.abspath(os.path.join(ROOTPATH, 'plots'))
sys.path.append(ROOTPATH)
sys.path.append(SOURCEPATH)
sys.path.append(PARAMSPATH)

# from user_params import cosmo_params, physics_units, PBHForm, Pk_models, verbose, MergingRates_models
from params.user_params import physics_units, cosmo_params, PSModels_params
from params.user_params import PBHFormation_params, MerginRates_params
from params.user_params import verbose 

PS_models = PSModels_params

# print(PS_models.model.gaussian)

# Base Class model 

class PS_Base:
    def __init__(self, As_cosmo=None, ns_cosmo=None, kstar_cosmo=None, cm=None): 

        cm = cm if cm else cosmo_params
        self.cm = cm
        self.As_cosmo = As_cosmo if As_cosmo else cm.As
        self.ns_cosmo = ns_cosmo if ns_cosmo else cm.ns
        self.kstar_cosmo = kstar_cosmo if kstar_cosmo else cm.kp
        # self.is_kp_cnst = True


    def PS_vac(self, kk):
        PS = self.As_cosmo * (kk / self.kstar_cosmo) ** (self.ns_cosmo - 1.) 
        return PS

    def PS(self, kk):
        # PS =  PS_vac(self, kk)
        print(">> Powerspectrum (PS) not specified, assuming PS vacuum.")
        return  self.PS_vac(kk)

    def PS_plus_vacuum(self, kk):
        return self.PS_vac(kk) + self.PS(kk)
    
    def ps_of_k(self, kk, with_vacuum=True):
        if with_vacuum: return self.PS_vac(kk) + self.PS(kk)
        else:  return self.PS_vac(kk)
   
    def get_children_strings(self):
        list_of_strings = []
        out = dict()
        for attr_name in dir(self):
            if attr_name not in dir(PS_Base):
                attr = getattr(self, attr_name)
                if hasattr(attr, 'get_children_strings'):
                    list_of_strings.extend(["." + attr_name + child_string for child_string in attr.get_children_strings()])
                else:
                    list_of_strings.append("" + attr_name + " = " + str(attr))
                    out["./" + attr_name] = attr
        return  list_of_strings 

    def get_attr(self):
        out = dict()
        for attr_name in dir(self):
            if attr_name not in dir(PS_Base()):
                attr = getattr(self, attr_name)
                out[attr_name] = attr
        return  out 

    def print_att(self):
        out = self.get_attr()
        print("Attributes of ", self.__class__.__name__ , "\n   >>   ", out)



#  Coded models:  

class PS_Vacuum(PS_Base):
    def __init__(self, As_cosmo=None, ns_cosmo=None, kstar_cosmo=None, cm=None): 
        super().__init__()
        cm = cm if cm else cosmo_params
        self.As = As_cosmo if As_cosmo else cm.As
        self.ns = ns_cosmo if ns_cosmo else cm.ns
        self.kstar = kstar_cosmo if kstar_cosmo else cm.kp

    def PS(self, kk):
        PS = self.As * (kk / self.kstar) ** (self.ns - 1.) 
        return PS


class PS_Powerlaw(PS_Base):
    
    def __init__(self, As=None, ns=None, kp=None,  ktrans=None, cm=None): 
        super().__init__()
        cm = cm if cm else cosmo_params
        self.As = As if As else PS_models.powerlaw.AsPBH
        self.ns = ns if ns else PS_models.powerlaw.nsPBH
        self.kp = kp if kp else PS_models.powerlaw.kp
        self.ktrans = ktrans if ktrans else PS_models.powerlaw.ktrans


    def PS(self, kk):
        PS =self.As * (kk / self.kp) ** (self.ns - 1) * np.heaviside(kk - self.ktrans, 0.5)
        return PS


class PS_LogNormal(PS_Base): 
    def __init__(self, As=None, sigma=None, kp=None, cm=None): 
        
        super().__init__()
        cm = cm if cm else cosmo_params
        self.As = As if As else PS_models.lognormal.AsPBH
        self.sigma = sigma if sigma else PS_models.lognormal.sigma
        self.kp = kp if kp else PS_models.lognormal.kp
    
    def PS(self, kk):
        
        PS = self.As * np.exp(- np.log(kk / self.kp) ** 2 / (2 * self.sigma ** 2))
        return PS


class PS_Gaussian(PS_Base):
    
    def __init__(self, As=None, sigma=None, kp=None, cm=None, verbose=False): 
        
        super().__init__()
        cm = cm if cm else cosmo_params
        self.As = As if As else PS_models.gaussian.AsPBH
        self.sigma = sigma if sigma else PS_models.gaussian.sigma
        self.kp = kp if kp else PS_models.gaussian.kp

        if verbose: print(f"Guassian PS loaded with {self.As}, {self.sigma}, {self.kp} ")
    
    
    def PS(self, kk):
        Pk = self.As * np.exp(- (kk - self.kp)** 2 / (2 * (self.sigma * self.kp) ** 2))
        # PS = self.AsPBH * np.exp(-(kk - self.kp)** 2 / (2 * (self.sigma) ** 2))
        return Pk


class PS_BrokenPowerlaw(PS_Base):

    def __init__(self, As_low=None, As_high=None, kp_low=None, kp_high=None, ns_low=None, ns_high=None, kc=None,cm=None): 
        
        super().__init__()
        cm = cm if cm else cosmo_params
        self.As_low = As_low if As_low else PS_models.broken_powerlaw.AsPBH_low
        self.As_high = As_high if As_high else PS_models.broken_powerlaw.AsPBH_high
        self.kp_low = kp_low if kp_low else PS_models.broken_powerlaw.kp_low
        self.kp_high = kp_high if kp_high else PS_models.broken_powerlaw.kp_high
        self.kc = kc if kc else PS_models.broken_powerlaw.kc
        self.ns_high = ns_high if ns_high else PS_models.broken_powerlaw.ns_high
        self.ns_low = ns_low if ns_low else PS_models.broken_powerlaw.ns_low


    def PS(self, kk):
        if isinstance(kk, float):
            kk = np.array([kk])

        PS = np.zeros_like(kk)
        mask = (kk < self.kc)
        PS[mask] = self.As_low * (kk[mask] / self.kp_low) ** self.ns_low
        PS[~mask] = self.As_high * (kk[~mask] / self.kp_high) ** self.ns_high
        return PS


class PS_AxionGauge(PS_Base):
    
    def __init__(self, As=None, sigma=None, kp=None, cm=None, with_vacuum=True): 
        
        super().__init__()
        cm = cm if cm else cosmo_params
        self.As = As if As else PS_models.axion_gauge.AsPBH
        self.sigma = sigma if sigma else PS_models.axion_gauge.sigma
        self.kp = kp if kp else PS_models.axion_gauge.kp
        self.with_vacuum = with_vacuum
    
    
    def PS_without_vacuum(self, kk):
        PS = self.As * np.exp(- np.log(kk / self.kp) ** 2 / (2 * self.sigma ** 2))
        return PS

    def PS(self, kk):
        if self.with_vacuum:
            return self.PS_vac(kk) + self.PS_without_vacuum(kk) 
        else:
            return self.PS_without_vacuum(kk) 
            # PS = self.As * np.exp(- np.log(kk / self.kp) ** 2 / (2 * self.sigma ** 2))
            # return PS


class PS_Preheating(PS_Base):

    
    def __init__(self, Hstar=None, e1=None, e2=None, C=None, kend=None, cm=None): 
        
        super().__init__()
        self.cm = cm if cm else cosmo_params
        self.Hstar = Hstar if Hstar else PS_models.preheating.Hstar
        self.e1 = e1 if e1 else PS_models.preheating.e1
        self.e2 = e2 if e2 else PS_models.preheating.e2
        self.C = C if C else PS_models.preheating.C
        self.kend = kend if kend else PS_models.preheating.kend
    
    
    def PS(self, kk):

        if isinstance(kk, float):  # necessary when using mask
            kk = np.array([kk])

        cm = self.cm
        pu = physics_units

        PS = np.zeros_like(kk)
        mask = (kk < self.kend)
        P0 = self.Hstar**2 / ( self.e1 *8 * np.pi**2 * pu.m_planck**2 )               #TODO : define Hstar as function of As??
        PS[~mask] = P0 
        PS[mask] = self.Hstar**2 / (8 * np.pi**2 * pu.m_planck**2  * self.e1) * \
                (1 + (kk[mask]/self.kend)**2) * (1 - 2*(self.C+1)*self.e1 - self.C * self.e2)
        
            # TODO: Hstar, e1 and e2  should be k-dependent ??.

        return PS


class PS_Multifield(PS_Base):
    
    def __init__(self, P0=None, eta=None, delta=None, kf=None): 
        
        super().__init__()
        self.P0 = P0 if P0 else PS_models.multifield.P0
        self.eta = eta if eta else PS_models.multifield.eta
        self.delta = delta if delta else PS_models.multifield.delta
        self.kf = kf if kf else PS_models.multifield.kf


    def PS(self, kk):

        if isinstance(kk, float):  # necessary when using mask
            kk = np.array([kk])

        PS = np.zeros_like(kk)
        kappa = kk/self.kf
        arg = (2-kappa)*kappa
        mask = (kappa < 1.7)

        PS[~mask] = self.P0
        PS[mask] = self.P0 * np.exp(2*np.sqrt(arg[mask])*self.eta*self.delta) / (2*arg[mask]) * \
            np.sin(np.exp(-self.delta/2)*kappa[mask]*self.eta + np.arctan(kappa[mask]/np.sqrt(arg[mask])))
        return PS 


class PS_UserImport(PS_Base):

    def __init__(self, user_k, user_PS): 
        self._user_k = user_k 
        self._user_PS = user_PS

        self.interp_func = interp1d(user_k, user_PS) 
            
    def PS(self, kk):
        PS = self.interp_func(kk)
        return PS


class PS_UserFunction(PS_Base):

    def __init__(self, func):
        self.myPSfunction = func
    

    def PS(self, kk):
        # returns the primordial power spectrum of curvature fluctuations at scale kk
        return self.myPSfunction(kk)


# Ensamble of models: 

class PowerSpectrum:

    gaussian = PS_Gaussian
    powerlaw = PS_Powerlaw
    lognormal = PS_LogNormal
    broken_powerlaw = PS_BrokenPowerlaw
    axion_gauge = PS_AxionGauge
    preheating = PS_Preheating
    multifield = PS_Multifield
    user_import = PS_UserImport
    user_function = PS_UserFunction
    vacuum = PS_Vacuum

    def get_defaultPS():
        if verbose > 2 : print("The default powerspectrum is Powerlaw.")
        return PS_Powerlaw

    default = get_defaultPS()

    def get_model(model, **kargs):

        if model=="gaussian": return PowerSpectrum.gaussian(kargs)
        if model=="powerlaw": return PowerSpectrum.powerlaw(kargs)
        if model=="lognormal": return PowerSpectrum.lognormal(kargs)
        if model=="broken_powerlaw": return PowerSpectrum.broken_powerlaw(kargs)
        if model=="axion_gauge": return PowerSpectrum.axion_gauge(kargs)
        if model=="preheating": return PowerSpectrum.preheating(kargs)
        if model=="multifield": return PowerSpectrum.multifield(kargs)
        if model=="vacuum": return PowerSpectrum.vacuum(kargs)


   






#######################################

if __name__ == "__main__":
    # Munch allows to set-up/extract values as  both dictionary and class ways

    # Example one model
    myPS = PowerSpectrum.gaussian()

    # myPS.As = 0.3
    myPS.print_att()

    # ks = 10**np.linspace(1, 5, 1000)
    ks = 10**np.linspace(1.0, 8.0, 100, True)   
    print( np.min(myPS.PS(ks)), np.max(myPS.PS(ks))  )


    import matplotlib.pyplot as plt

    plt.plot(ks, myPS.PS(ks))
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(1e-19, 1)
    plt.savefig(PLOTSPATH + "/example_powerspectra_gaussian.png")
    # plt.show()
    plt.close()

    xmin = 10**2
    xmax = 10**13
    ymin = 1e-12
    ymax = 1.05
    k_values = 10**np.linspace(np.log10(xmin), np.log10(xmax), 200)
    


    # Example with several models
    
    models = [ 
        'powerlaw',
        'broken_powerlaw',
        'lognormal',
        'gaussian',
        'multifield',
        'axion_gauge',
        # 'preheating',
        'vacuum'
    ]

    model_name = [
        'power-law',
        'broken power-law',
        'lognormal',
        'gaussian',
        'multifield',
        'axion gauge',
        # 'preheating',   #   (2.20)
        'vacuum'
    ]

    color_pal = ['k', 'b', 'g', 'r',  'orange', 'darkgreen', 'purple', 'k']
    lstyle = ['-', '-', '-', '-',      '--', '--', '--', '--']

    figPk = plt.figure()
    figPk.patch.set_facecolor('white')
    ax = figPk.add_subplot(111)
    
    for i, model in enumerate(models):
        PM = PowerSpectrum.get_model(model)
        xs = k_values
        ys = PM.PS(kk=k_values)     
        lbl = "{}".format(model_name[i])
        ax.plot(xs, ys, label=lbl, color=color_pal[i], ls=lstyle[i])
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.ylim(ymin,ymax)
    plt.xlim(xmin,xmax)
    plt.xlabel('Wavenumber $k$  [Mpc$^{-1}$]', fontsize=14)
    plt.ylabel(r'$\mathcal{P}_{\zeta} (k)$', fontsize=15)
    
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTSPATH + "/example_powerspectra_models.png")
    plt.show()

    
