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

# from user_params import cosmo_params, physics_units, PBHForm
from params.user_params import physics_units, cosmo_params, PSModels_params
from params.user_params import PBHFormation_params, MerginRates_params
from params.user_params import verbose 

PBHForm = PBHFormation_params


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
                print(f"!!! Thermal history file : file path not found \n ? {fid_file_path}")

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
                       ps_function, 
                       eta=PBHForm.models.Musco20.eta,                      
                       k_star=PBHForm.models.Musco20.k_star,                      
                       ps_scalefactor=PBHForm.Pkscalingfactor,
                       force_method=False,
                       pm=False, cp=False, thermalhistory_func=False ):
        super().__init__()
        self.eta = eta
        self.ps_function= ps_function
        self.k_star = k_star
        self.ps_scalingfactor = ps_scalefactor
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
        Pk = self.ps_function 

        P = Pk(k * self.k_star) * self.ps_scalingfactor
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

            return  ClassPBHFormationStandard(self.ps_function).get_deltacr()
        
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
    # PS_func =  PS_model.PS_plus_vacuum         # This is the default to calculate sigma and fPBH
        
    PS_model = PowerSpectrum.axion_gauge()    
    PS_func =  PS_model.PS_plus_vacuum


    ### Print threshold 

    print("\n")
    print("Example using Standard formalism:  ")
    deltacrit = ClassPBHFormationStandard(PS_func=PS_func)

    dc = deltacrit.get_deltacr()
    dc_thermal = deltacrit.get_deltacr_with_thermalhistory(mPBH)
    print(" >> delta crit without / with thermal history ", dc, dc_thermal)

    print("\n")
    print("Example using Musco formalism: ")
    deltacrit = ClassPBHFormationMusco20(ps_function=PS_func)

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