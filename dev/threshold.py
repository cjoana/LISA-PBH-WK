
import numpy as np
import scipy.constants as const
import scipy.special as special
from scipy.special import erfc
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.integrate import dblquad
import scipy.optimize as opt


import sys, os
FILEPATH = os.path.realpath(__file__)[:-17]
sys.path.append(FILEPATH + "/src")
sys.path.append("./src")
print(f"FILEPATH = {FILEPATH}")

from user_params import cosmo_params, physics_units, PBHForm


verbose = 0


class ClassDeltaCritical: 

    def __init__(self):
        self.pm=PBHForm
        self.cp=cosmo_params
        # self.thermalhistory_func = self.get_thermalfactor
        self.thermalhistory_func = self.get_thermalfactor_from_file


    def get_deltacr():
        print("!!! delta critical is not set")
        raise 

    # def get_thermalfactor(self, mPBH, **kargs):
    #     # if self.get_th_from_file: 
    #     return self.get_thermalfactor_from_file(mPBH, **kargs)


    def get_thermalfactor_from_file(self, mPBH, datadir=None, thermal_file=None, zetacr_thermal_rad=None):

        datadir = datadir if datadir else  self.pm.data_directory
        thermal_file = thermal_file if thermal_file else self.pm.zetacr_thermal_file
        zetacr_thermal_rad = zetacr_thermal_rad if zetacr_thermal_rad else self.pm.zetacr_thermal_rad

        # Read file
        def _read_thermal_file(datadir, thermal_file):     #TODO (put as a separete class?)   

            # returns a vector of zeta_cr and gthe corresponding vector of m_PBHs
            Npointsinfile = 1501 + 11                                        #TODO: REDO for any file!!! 
            log_m_in_file = np.zeros(Npointsinfile)
            zetacr_in_file = np.zeros(Npointsinfile)
            # Read file of zetacr_thermal as a function of
            fid_file_path = os.path.join(datadir, thermal_file)
            if verbose > 2 : print("I use the following file for thermal history: ", fid_file_path)
            if os.path.exists(fid_file_path):
                fid_values_exist = True
                with open(fid_file_path, 'r') as fid_file:
                    line = fid_file.readline()
                    while line.find('#') != -1:
                        line = fid_file.readline()
                    while (line.find('\n') != -1 and len(line) == 1):
                        line = fid_file.readline()
                    for index_mass in range(Npointsinfile):
                        logmPBH = np.log10(float(line.split()[0]))
                        log_m_in_file[index_mass] = logmPBH  # np.log10(double(line.split()[0])
                        zetacr_in_file[index_mass] = float(line.split()[1])
                        line = fid_file.readline()
            else:
                print("!!! Thermal history file : file path not found")

            return log_m_in_file, zetacr_in_file
        
        log_m_in_file, zetacr_in_file = _read_thermal_file(datadir, thermal_file)
        # Returns the factor from thermal history by which one has to multiply the zeta_cr obtained for radiation
        zetacr_interp = interp1d(log_m_in_file, zetacr_in_file, kind='linear')  
        logmPBH = np.log10(mPBH)
        thermal_factor = zetacr_interp(logmPBH) / zetacr_thermal_rad
        return thermal_factor


    def get_deltacr_with_thermalhistory(self, mPBH):

        deltacr_rad = self.get_deltacr()
        deltacr_with_th =  deltacr_rad * self.thermalhistory_func(mPBH)

        return deltacr_with_th
    

class ClassPBHFormationMusco20(ClassDeltaCritical):

    def __init__(self, 
                       PS_func, 
                       eta=PBHForm.models.Musco20.eta,                      
                       k_star=PBHForm.models.Musco20.k_star,                      
                       Pk_scalefactor=PBHForm.Pkscalingfactor,
                       force_method=False,
                       pm=False, cp=False, thermalhistory_func=False ):
        super().__init__()
        self.eta = eta
        self.PS_func= PS_func
        self.k_star = k_star
        self.Pkscalingfactor = Pk_scalefactor
        self.force_method = force_method
        if pm: self.pm = pm
        if cp: self.cp = cp
        if thermalhistory_func: self.thermalhistory_func

        # if verbose >2: print( 'FormPBHMusco (class) set  Pk to ', Pk_model)

    # List of fnctions for the computations of delta_critical based on the method Musco20
    def TransferFunction(self, k, t):
        sq = 1. / np.sqrt(3)
        arg = k * t * sq
        return 3 * (np.sin(arg) - arg * np.cos(arg)) / arg ** 3

    def PowerSpectrum(self, k, t):
        Pk = self.PS_func 

        P = Pk(k * self.k_star) * self.Pkscalingfactor
        T =  self.TransferFunction(k, t)
        return 2.*np.pi**2 /(k**3) * P * T**2

    def ShapeRHS(self, t, rm=1.0, print_errors=False):
        cos = lambda k: (k ** 4 * np.cos(k * rm) * self.PowerSpectrum(k, t))
        sin = lambda k: (k ** 3 * np.sin(k * rm) * self.PowerSpectrum(k, t))
        cosint = integrate.quad(cos, 0, np.inf, limit=100000, limlst=10000)
        sinint = integrate.quad(sin, 0, np.inf, limit=100000, limlst=10000)
        coserr = cosint[1]
        sinerr = sinint[1]

        if print_errors:
            print("errs = ", coserr, sinerr)

        result = -0.25 * (1 + rm * cosint[0] / sinint[0])
        return result

    def F_alpha(self, a):
        
        arg = 5 / (2 * a)

        # gammainc  ==  inc lower gammar   Normalized  (1/gamma(arg))
        # gammaincc ==  inc upper gammar  Normalized (1/gamma(arg))
        diff = (special.gamma(arg) * special.gammainc(arg, 1 / a))

        if not diff==diff:
            print('diff = ', diff,
                  " \n a=5/(2*alpha), x = 1/alpha , G(a), G_inc(a, x) = ",
                  arg, 1/a, special.gamma(arg),  special.gammainc(arg, 1 / a),  "\n\n" )
            raise Exception('  !!  nan at calc Gammas in F(alpha)')

        disc = 1 - 2 / 5 * np.exp(-1 / a) * a ** (1 - arg) / diff

        if disc < 0 or not disc==disc:
            print('disc = ', disc, "\n\n" )
            raise Exception('  !!  negative  sqrt  or nan in F(alpha)')

        return np.sqrt(disc)

    def get_rm(self, t=None,  guess=1.0, method='root'):

        t = t if t else self.eta

        def func(rm):
            integrand = lambda k: k ** 2 * (
                        (k ** 2 * rm ** 2 - 1) * np.sin(k * rm) / (k * rm) + np.cos(k * rm)) \
                                  * self.PowerSpectrum(k, t)
            integ = integrate.quad(integrand, 0, np.inf, limit=1000, limlst=100)
            # integ = integrate.quad(integrand, 0, 1e10, limit=100000, limlst=10000)  #TODO: why it crashes without inf
            return integ[0]

        if method == 'root':
            sol = opt.root(func, x0=guess)
            root = sol.x
            success = sol.success

        else:
            root, success = opt.bisect(func, 0., 1e8, rtol=1.e-5, maxiter=100, full_output = True)
            # success = True

        if success:
                return float(root)
        else:
                raise Exception("failed to converge in get_rm iteration")

    def ShapeValue(self, t, rm=1.0, guess=0.5, method='root'):

        def func(a):
            return self.F_alpha(a) * (1 + self.F_alpha(a)) * a - 2 * self.ShapeRHS(t, rm=rm)

        # method = 'bisect' #TODO

        if method == 'root':
            sol = opt.root(func, x0=guess)
            root = sol.x
            success = sol.success
        else:
            root, success = opt.bisect(func, 0.01, 100., rtol=1.e-5, maxiter=100, full_output = True)

        if success:
            return float(root)
        else:
            raise Exception("failed to converge in ShapeValue iteration")

    def dcrit(self, a):

        if (a >= 0.1 and a <= 3):
            return a ** 0.125 - 0.05
        if (a > 3 and a <= 8):
            return a ** 0.06 + 0.025
        if (a > 8 and a <= 30):
            return 1.15
        
        else: 
            err_msg = f"\n!!! the value of alpha is out of the allowed window (0.1, 30),\n alpha = {a}\n"
            print(err_msg)
            if self.force_method:  
                raise Exception(err_msg)

            return  ClassPBHFormationStandard(self.PS_func).get_deltacr()
        
    def get_deltacr(self):
        # print("getting delta_cr with eta =  ", self.eta,  " kp =", self.kp) #TODO :clean
        
        eta = self.eta       
        if verbose: print(f" we found eta = {eta}")

        rm = self.get_rm(eta)
        if verbose: print(f" we found rm = {rm}")

        alpha = self.ShapeValue(eta, rm=rm)                 #TODO: large alpha values (>30) crashes the code
        if verbose: print(f" we found alpha = {alpha}")

        deltacr = self.dcrit(alpha)
        if verbose: print(f" we found deltacr = {deltacr}")

        return deltacr


class ClassPBHFormationStandard(ClassDeltaCritical):

    def __init__(self, PS_func, pm=False, cp=False, thermalhistory_func=False):
        super().__init__()
        if pm: self.pm = self.pm=pm
        if cp: self.cp = cp
        if thermalhistory_func: self.thermalhistory_func = thermalhistory_func
        self.PS_func=PS_func

    
    def get_deltacr(self):

        deltacr_rad =self.pm.models.standard.deltacr_rad
        return deltacr_rad

class ClassThresholds:
    standard = ClassPBHFormationStandard
    Musco20 = ClassPBHFormationMusco20


#### TODO:   WORK IN PROGRESS

class __deprecated__ClassPBHFormationStandard:

    def __init__(self, pm=PBHForm, cp=cosmo_params, Pk_model='default', PBHform_model='default'):
        if PBHform_model == 'default':
            PBHform_model = pm.PBHform_model
        self.pm = pm
        self.cp = cp
        self.ratio_mPBH_over_mH = pm.ratio_mPBH_over_mH
        self.kmsun = pm.kmsun
        self.Pkscalingfactor = pm.Pkscalingfactor
        self.use_thermal_history = pm.use_thermal_history
        self.Pkrescaling = pm.Pkrescaling
        self.rescaling_is_done = False
        self.forcedfPBH = pm.forcedfPBH
        self.use_thermal_history = pm.use_thermal_history
        self.data_directory = pm.data_directory
        self.zetacr_thermal_file = pm.zetacr_thermal_file
        self.zetacr_thermal_rad = pm.zetacr_thermal_rad
        self.Gaussian = pm.Gaussian
        self.Pk_model = Pk_model
        self.PBHform_model = pm.models[PBHform_model].PBHform_model

        if verbose:
            print('FormPBH (class) set  PBHform_model to ', self.PBHform_model)
            print('FormPBH (class) set  Pk to ', Pk_model)


    def get_thermalfactor(self, mPBH):
        # Read file
        def _read_thermal_file(datadir, zetacr_path):     #TODO (put as a separete class?)

            # returns a vector of zeta_cr and gthe corresponding vector of m_PBHs
            Npointsinfile = 1501 + 11
            log_m_in_file = np.zeros(Npointsinfile)
            zetacr_in_file = np.zeros(Npointsinfile)
            # Read file of zetacr_thermal as a function of
            fid_file_path = os.path.join(datadir, zetacr_path)
            if verbose > 2 : print("I use the following file for thermal history: ", fid_file_path)
            if os.path.exists(fid_file_path):
                fid_values_exist = True
                with open(fid_file_path, 'r') as fid_file:
                    line = fid_file.readline()
                    while line.find('#') != -1:
                        line = fid_file.readline()
                    while (line.find('\n') != -1 and len(line) == 1):
                        line = fid_file.readline()
                    for index_mass in range(Npointsinfile):
                        logmPBH = np.log10(float(line.split()[0]))
                        log_m_in_file[index_mass] = logmPBH  # np.log10(double(line.split()[0])
                        zetacr_in_file[index_mass] = float(line.split()[1])
                        line = fid_file.readline()

            return log_m_in_file, zetacr_in_file

        log_m_in_file, zetacr_in_file = _read_thermal_file(self.data_directory, self.zetacr_thermal_file)
        # Returns the factor from thermal history by which one has to multiply the zeta_cr obtained for radiation
        zetacr_interp = interp1d(log_m_in_file, zetacr_in_file, kind='linear')  # , kind='cubic')
        logmPBH = np.log10(mPBH)
        thermal_factor = zetacr_interp(logmPBH) / self.zetacr_thermal_rad
        return thermal_factor

    def get_deltacr(self, method='default'):


        if (method == 'default'):
            method = self.PBHform_model

        # Depends method
        if (method=='standard'):
            deltacr_rad =self.pm.models.standard.zetacr_rad
        elif (method == "Musco20"):
            pmm = ClassPBHFormationMusco20(Pk_model=self.Pk_model, cp=self.cp)
            deltacr_rad = pmm.get_deltacr()
        else:
            mess = 'Such method ({m}) is still not implemented yet'.format(m=method)
            print(mess)
            raise Exception(mess)

        return deltacr_rad

    def get_zetacr(self, mPBH, method='default', use_thermal_history='default'):

        if (method == 'default'):
            method = self.PBHform_model
        if (use_thermal_history == 'default'):
            use_thermal_history = self.use_thermal_history

        if self.Pkrescaling == True and self.rescaling_is_done == False:
            self.calc_scaling(mPBH)

        zetacr_rad = self.get_deltacr(method=method)

        # Usage of thermal history
        if (use_thermal_history == True):
            zetacrtable = zetacr_rad * self.get_thermalfactor(mPBH)
        else:
            zetacrtable = zetacr_rad

        return zetacrtable

    ###############################################################################################

    def get_logbeta(self, mPBH, use_thermal_history='default'):  #TODO

        if use_thermal_history == 'default':
            use_thermal_history = self.use_thermal_history

        if self.Pkrescaling == True and self.rescaling_is_done == False:
            self.calc_scaling(mPBH)

        PkS = ClassPkSpectrum(pkmodel=self.Pk_model, cm=self.cp)

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
        return ClassPkSpectrum(pkmodel=self.Pk_model, cm=self.cp).Pk(k)



#######################################

if __name__ == "__main__":

    from power_spectrum import PowerSpectrum

    ###### Set mass example:

    Msun = physics_units.m_sun
    # mPBH = Msun * 1.0
    mPBH = 1.0

    ###### Set model example: 

    # #TODO: Gaussian model leads to large alpha values >30, Musco method do not work 
    # sig = 0.25 
    # As = 0.01*sig
    # kp = 1e7  # TODO: with 1e6 or 1e8 it crashes!!
    # PS_model = PowerSpectrum.gaussian(As=As, sigma=sig, kp=kp)
    # PS_func =  PS_model.PS_plus_vaccumm         # This is the default to calculate sigma and fPBH
        
    PS_model = PowerSpectrum.axion_gauge()    
    PS_func =  PS_model.PS_plus_vaccumm


    ### Print threshold 

    print("\n")
    print("Example using Standard formalism:  ")
    deltacrit = ClassPBHFormationStandard(PS_func=PS_func)

    dc = deltacrit.get_deltacr()
    dc_thermal = deltacrit.get_deltacr_with_thermalhistory(mPBH)
    print(" >> delta crit without / with thermal history ", dc, dc_thermal)

    print("\n")
    print("Example using Musco formalism: ")
    deltacrit = ClassPBHFormationMusco20(PS_func=PS_func)

    dc = deltacrit.get_deltacr()
    dc_thermal = deltacrit.get_deltacr_with_thermalhistory(mPBH)
    print(" >> delta crit without / with thermal history ", dc, dc_thermal)


    # ###  Example by seting own thermal function: 

    # def th_func(*args): return 1
    # thermalhistory_func = th_func    
    # print("\n")
    # print("Example using Standard formalism (own thermal func.):  ")
    # deltacrit = ClassPBHFormationStandard(PS_func=PS_func, thermalhistory_func=thermalhistory_func )
    # dc = deltacrit.get_deltacr()
    # dc_thermal = deltacrit.get_deltacr_with_thermalhistory(mPBH)
    # print(" >> delta crit without / with thermal history ", dc, dc_thermal)
    # print("\n")
    # print("Example using Musco formalism: ")
    # deltacrit = ClassPBHFormationMusco20(PS_func=PS_func, thermalhistory_func=thermalhistory_func)
    # mPBH = 1.0
    # dc = deltacrit.get_deltacr()
    # dc_thermal = deltacrit.get_deltacr_with_thermalhistory(mPBH)
    # print(" >> delta crit without / with thermal history ", dc, dc_thermal)