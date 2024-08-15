"""
Code to evaluate SGWB originated by PBH mergers.  Code provided by Eleni and Satchiko (check?!)
"""

import sys, os
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
SOURCEPATH = os.path.abspath(os.path.join(ROOTPATH, 'source'))
PARAMSPATH = os.path.abspath(os.path.join(ROOTPATH, 'params'))
PLOTSPATH = os.path.abspath(os.path.join(ROOTPATH, 'plots'))
sys.path.append(ROOTPATH)
sys.path.append(SOURCEPATH)
sys.path.append(PARAMSPATH)


from abundances import CLASSabundances
from user_params import cosmo_params, physics_units
from default_params import p_PhysicsUnits as pu
from default_params import p_CosmologicalParameters as cp
from merger_rates import MergerRates
from power_spectrum import PowerSpectrum
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import numpy as np
import scipy

class Backgrounds(MergerRates):
    def __init__(self,abundances=None,zmin=0,zmax=100,mPBHmin=1.,mPBHmax=10.):
        
        if not abundances:
            raise ValueError("Abundances has not been specified in merger_SGWB class.")
        self.my_abundances=abundances
        self.zmin = zmin
        self.zmax = zmax
        self.logm1min = np.log10(mPBHmin)
        self.logm2min = np.log10(mPBHmin)
        self.logm1max = np.log10(mPBHmax)
        self.logm2max = np.log10(mPBHmax)

        self.fpbh_integrated = 1  # TODO: this need to be computed

#    def make_t_of_z(self):        
#        N = 200   # making a list of redshifts z from 0 to 100 in logspace, with small step(before : N = 1000, z_min = 1e-30)
#        z_min = 1e-4 
#        redsh = np.logspace(np.log10(z_min),2,N)
#        tofz = []    # creating t(z) by integrating the function "argt(z) from i to 10000"
#        for i in redsh:
#            tofz.append(quad(arg_t,i,10000,epsabs=0,epsrel=1e-4)[0])    
#        t_interp_log = interpolate.interp1d(np.log(redsh),np.log(tofz), kind = 'linear') # interpolation of tofz
#        self.t_interp = lambda z: np.exp(t_interp_log(np.log(z)))
#        return 

    def _Get_GW_bkg_single_freq(self, freq, rate_model):
        
        # Computation of the GW background in a m_1, m_2 grid
        
        # Computation of the total GW background
        def integrant(logm1,logm2,z):
            mc53 = 10.**logm1 * 10.**logm2 /(10.**logm1 + 10.**logm2)**(1./3.) * (pu().m_sun)**(5./3.)
            Hofz = cp().H0 * np.sqrt( (cp().Omb + cp().Omc) * (1+z)**3 + cp().OmLambda) 
            z_dep = (Hofz * (1+z)**(4/3))**(-1)
            fPBH1 = self.my_abundances.get_fPBH(10.**logm1)
            fPBH2 = self.my_abundances.get_fPBH(10.**logm2)
            fisco = 4400./(10.**logm1+10.**logm2)
            if freq > fisco:
                integrant = 0.
            else:
                integrant = freq**(-4./3.) *(4. * pu().G**(5./3.)) / (3. * np.pi**(1./.3)\
                             * pu().c**2) * mc53 * z_dep \
                             * rate_model(self.fpbh_integrated, 10.**logm1,10.**logm2,fPBH1,fPBH2) / pu().year / ((1.E3*pu().mpc)**3) 
            
            return integrant

        #print("coucou")
        #print(integrant(0,0,1) * np.pi /(4.*pu().G) * freq**2 / cp().rhocr)
             
        #hc2 = scipy.integrate.tplquad(integrant, self.zmin, self.zmax, self.logm2min, self.logm2max, self.logm1min, self.logm1max, epsrel=1.e-1)[0] 

        def z_integrant(z):
            Hofz = cp().H0 * np.sqrt( (cp().Omb + cp().Omc) * (1+z)**3 + cp().OmLambda) 
            z_dep = (Hofz * (1+z)**(4/3))**(-1)
            return z_dep

        integrant2 = lambda logm1, logm2: integrant(logm1, logm2, 0.)
        zintegral = scipy.integrate.quad(z_integrant,self.zmin, self.zmax, epsrel=1.e-2) 
        hc2 = scipy.integrate.dblquad(integrant2, self.logm2min, self.logm2max, self.logm1min, self.logm1max, epsrel=1.e-6)[0]
        hc2 *= zintegral[0]/z_integrant(0)    
        GWh2_bkg= np.pi /(4.*pu().G) * freq**2 *hc2 / cp().rhocr * cp().h**2
        
        return GWh2_bkg


    def Get_GW_bkg(self, freq, rate_model):
        
        if isinstance(freq, float):
            GWh2_bkg =  self._Get_GW_bkg_single_freq(freq, rate_model)
        else:
            GWh2_bkg = np.zeros_like(freq)
            for i_freq, single_freq in enumerate(freq):
                GWh2_bkg[i_freq] = self._Get_GW_bkg_single_freq(single_freq, rate_model)

        return GWh2_bkg


    def Get_GW_bkg_primordial_binary(self, freq):    
        
        GW_bkg = self.Get_GW_bkg(freq, MergerRates().rates_early_binaries)
        #print('gw_bkg_primordial: ', GW_bkg)
        return GW_bkg
        
    def Get_GW_bkg_cluster_binary(self, freq):
        GW_bkg = self.Get_GW_bkg(freq, MergerRates().rates_late_binaries)
        #print('gw_bkg_cluster: ', GW_bkg)
        return GW_bkg
 
#    def Get_GW_bkg_from_Enc(self):
#        return GW_bkg


if __name__ == "__main__":

    masses =  10**np.linspace(-3,4, 100)  

    # PS_model = PowerSpectrum.gaussian(kp=2.e6, As=0.0205, sigma=1.)
    # PS_func =  PS_model.PS
    
    def PS_func(kk):
        AsPBH, kp, sigma = [0.0025, 2.e6, 1.]
        # AsPBH *= 1.183767
        return AsPBH * np.exp(- np.log(kk / kp) ** 2 / (2 * sigma ** 2))

    my_abundances = CLASSabundances(ps_function=PS_func)
    fpbhs = my_abundances.get_fPBH(masses)
    fpbh_integrated = 1


    sol = MergerRates().get_rates_late_binaries(fpbh_integrated, masses, fpbhs)
    my_backgrounds = Backgrounds(my_abundances)

    logfmin=-10
    logfmax=4
    nfreq=20
    GWB_EB=np.zeros(nfreq)
    GWB_LB=np.zeros(nfreq)
    listfreq =  np.logspace(logfmin,logfmax,nfreq)
    
    # for ifreq in range(nfreq):
    #     freq= listfreq[ifreq]
    #     # print(ifreq,freq)
    #     GWB_EB[ifreq] = my_backgrounds.Get_GW_bkg_primordial_binary(freq)
    #     GWB_LB[ifreq] = my_backgrounds.Get_GW_bkg_cluster_binary(freq)        

    GWB_EB = my_backgrounds.Get_GW_bkg_primordial_binary(listfreq)
    GWB_LB = my_backgrounds.Get_GW_bkg_cluster_binary(listfreq)  

    fig, ax = plt.subplots(1,1, figsize=(6, 5)) 

    ax.tick_params(axis='both', which='both', labelsize=11, direction='in', width=0.5) 
    ax.xaxis.set_ticks_position('both') 
    ax.yaxis.set_ticks_position('both') 
    for axis in ['top','bottom','left','right']: 
        ax.spines[axis].set_linewidth(0.5)
    plt.loglog(listfreq, GWB_EB, linestyle= 'solid', label = 'Total EB',color = '#010fcc')
    plt.loglog(listfreq, GWB_LB, linestyle= 'dashed', label = 'Total LB',color = 'red')    
    plt.xlabel(r"$f \thinspace\rm{(Hz)}$",fontsize = 14)
    plt.ylabel(r"$\Omega_{\rm{GW}} h^2$", fontsize = 14)
    plt.show()

