import numpy as np
from scipy import special
from scipy import integrate
from scipy.integrate import quad, dblquad, tplquad, simpson
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import math
import cmath
# f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
# g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
# fmt = mticker.FuncFormatter(g)
from matplotlib import rc
rc('text', usetex=True)

#Plotting Configurations
def fmt(x):
    root = x/10**(np.log10(x))
    exp = int(np.log10(x))
    out =  r"${r:.2f}$".format(r=root) + f'$\\times 10^{{{exp}}}$ '
    return out

m_Pl = 1.22*10.**(19.) # The Planck mass
M_Pl = m_Pl/np.sqrt(8.*np.pi) # The reduced Planck mass
gstar = 108. # The effective number of relativistic degrees of freedom above 10GeV
cg = 0.4
cs = 1./np.sqrt(3.) # The sound speed in the the radiation era
Omega_r_0 = 4.2*10.**(-5.) # the present day density of radiation
M_sun = 1.12*10.**(57.) #The mass of the sun in GeV
rho_0 = 1.04*(10**(-120))*(M_Pl**(4)) # The present day energy density
H_0 = np.sqrt(rho_0/3.)/M_Pl # The present day Hubble parameter
gam = 0.2 # the fraction of the cosmological horizon collapsing to a black hole

c_grams_to_GeV = 5.6*(10.**(23.))  # Conversion factor from grams to GeV
c_GeV_to_Mpc_minus_1 = 1.5637*10.**(38)   # Conversion factor from GeV to Mpc^{-1}
c_Hz_to_GeV = (6.58*((10.)**(-25.))) # Conversion factor from Hz to GeV


def k_f(M,Omega_PBH_f):
    return (((3.8*np.pi*gstar/480.)*(M_Pl**2.)/(2.*np.pi*gam*Omega_PBH_f**2.)/M**2.)**(1./6.))*(Omega_r_0**(1./4.))*((H_0*4.*np.pi*gam*(M_Pl**2.)/M)**(1./2.))*c_GeV_to_Mpc_minus_1

def k_UV(M,Omega_PBH_f):
    return k_f(M,Omega_PBH_f)*(Omega_PBH_f**(1./3.))*(gam)**(-1./3.)

def k_d(M,Omega_PBH_f):
    return np.sqrt(2.)*Omega_PBH_f*k_f(M,Omega_PBH_f)

def k_evap(M,Omega_PBH_f):
    return ((Omega_PBH_f*3.8*np.pi*gstar/(960.*np.pi*gam))**(1./3.))*((M/M_Pl)**(-2./3.))*k_f(M,Omega_PBH_f)

def integrand_SIGW_PBH_Poisson(s):
    value = ((1. - s**2.)**2.)/(1.-(cs**2.)*(s**2.))**(5./3.)
    return value

def s0(k,M,Omega_PBH_f):
	if k_UV(M,Omega_PBH_f)/k >= (1. + 1./cs)/2.:
		value = 1.
	elif (1./cs)/2.<= k_UV(M,Omega_PBH_f)/k <=(1. + 1./cs)/2.:
		value = 2.*k_UV(M,Omega_PBH_f)/k - 1./cs
	else:
		value = 0.
	return value

def Omega_SIGW_PBH_Poisson(k,M,Omega_PBH_f):

    # # The IR tail
    value_IR = cg*Omega_r_0*(9.*((3./2.)**(2./3.))*(cs**(4.))/(5120.*(np.pi**2.)))*((k_evap(M,Omega_PBH_f)/k_UV(M,Omega_PBH_f))**(10./3.))*((k_d(M,Omega_PBH_f)/k_evap(M,Omega_PBH_f))**8.)*(k/k_UV(M,Omega_PBH_f))


    # # The UV resonance contribution
    I = quad(integrand_SIGW_PBH_Poisson,-s0(k,M,Omega_PBH_f),s0(k,M,Omega_PBH_f))
    value_UV = cg*Omega_r_0*3.*(cs**(7./3.))*((1.-cs**2.)**2.)*((9./2.)**(1./3.))*((k/k_evap(M,Omega_PBH_f))**(11./3.))*((k_UV(M,Omega_PBH_f)/k_evap(M,Omega_PBH_f))**(2.))*((k_d(M,Omega_PBH_f)/k_UV(M,Omega_PBH_f))**8.)*I[0]/(np.pi*(2.**(14.)))

    value = ((3./2.)**4.)*(value_IR + value_UV)

    return value



# Choose here a value for the PBH mass and the initital PBH abundnace at formation time.
M_val = 5.*(10.**(7.))*c_grams_to_GeV
Omega_PBH_f_val = 2.*10.**(-9.)

k_range = np.linspace(np.log(k_evap(M_val,Omega_PBH_f_val)),np.log(1.2*k_UV(M_val,Omega_PBH_f_val)),200)
k_range = np.exp(k_range)

f_range  = (k_range/(2.*np.pi))/c_GeV_to_Mpc_minus_1/c_Hz_to_GeV

OmegaSIGW = np.array([Omega_SIGW_PBH_Poisson(k,M_val,Omega_PBH_f_val) for k in k_range])


# plotting 

plt.plot(f_range, OmegaSIGW,'b-',label = r'$\mathrm{SIGWs\; from\; the\; PBH\; Poisson\; Gas}$')
plt.xlim(xmin= 8.*10.**(-10.),xmax= 10.**(4.))
plt.ylim(ymin=10.**(-20.),ymax=8.*10.**(-6.))
plt.title(r'$M=5\times 10^{7}\mathrm{g}$,$\Omega_\mathrm{PBH,f}=2\times 10^{-9}$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$f(\mathrm{Hz})$',fontsize = 12)
plt.ylabel(r'$\Omega_{\mathrm{GW}}(\eta_\mathrm{0},k)h^2$',fontsize = 12)
plt.axhline(y = 6.9*10.**(-6.),label = r'$\Delta N_\mathrm{eff}\quad\mathrm{bound}$',color='k',linestyle='-')
plt.legend(loc = 'lower right',fontsize = 8)
plt.show()
