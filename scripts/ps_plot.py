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
#ROOTPATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
ROOTPATH = os.getcwd()
SOURCEPATH = os.path.abspath(os.path.join(ROOTPATH, 'dev'))
PARAMSPATH = os.path.abspath(os.path.join(ROOTPATH, 'params'))
PLOTSPATH = os.path.abspath(os.path.join(ROOTPATH, 'plots'))
sys.path.append(ROOTPATH)
sys.path.append(SOURCEPATH)
sys.path.append(PARAMSPATH)


# from baseclass import CLASSBase
# from power_spectrum import PowerSpectrum
# from threshold import ClassThresholds
# from merger_rates import MergerRates
# from abundances import CLASSabundances
from primbholes import primbholes 



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


from power_spectrum import PowerSpectrum

ComputeFPBH = True

# Example with several models
models = [ 
    # 'powerlaw',      # OK  
    # 'broken_powerlaw', # OK
    # 'lognormal',
    # 'gaussian',    # OK
    'multifield',
    # 'axion_gauge',
    # 'preheating',
    # 'vacuum'
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


model_name = models

color_pal = ['k', 'b', 'g', 'r',  'orange', 'darkgreen', 'purple', 'k']
lstyle = ['-', '-', '-', '-',      '--', '--', '--', '--']


xmin = 10**2
xmax = 10**13
ymin = 1e-12
ymax = 1.05
k_values = 10**np.linspace(np.log10(xmin), np.log10(xmax), 200)

figPk = plt.figure()
figPk.patch.set_facecolor('white')
ax = figPk.add_subplot(111)

for i, model in enumerate(models):
    PM = PowerSpectrum.get_model(model)
    ps_function = PM.PS
    
    ifpbh = ""
    if ComputeFPBH:
        pb = primbholes(ps_function=ps_function, fpbh_rescaling=1.0)
        ifpbh = np.round(pb.get_integrated_fPBH(), 2)
        scale = pb.get_spectrum_scaling()
    
    xs = k_values
    ys = PM.PS(kk=k_values)   * scale  
    lbl = f"{model_name[i]} -- fpbh = {ifpbh}, scale: {np.round(scale,3)}"
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
# plt.savefig(PLOTSPATH + "/example_powerspectra_models.png")
plt.show()


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


a = pb
# a = CLASSabundances(PS_function = PS_func, threshold_method="Musco20")

## Params range: 
# mass = 10**np.linspace(-10,20, 1000)  #* Msun
mass = 10**np.linspace(-6,8, 500)  #* Msun

floor = 1e-8
beta = a.get_beta(mass)  #+ floor
fpbh = a.get_fPBH(mass)  + floor
sigma = a.get_variance(mass)
fpbh_integrated =  a.get_integrated_fPBH()

fig, axs = plt.subplots(4,1, figsize=(8,8))

fig.suptitle("Example of Guassian powerspectra")
kk =  k_of_m(mass)
ax = axs[0]
ax.plot(kk, pb.ps_of_k(kk))
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
ax.set_xlabel(r"$m_{PBH}\ [M_{\rm sun}]$")
ax.set_xlim(max(mass), min(mass))
ax.set_ylim(beta.max()*1e-8, beta.max()*10)

ax = axs[3]
mss = r'integrated $f_{pbh}$ = ' + f'{np.round(fpbh_integrated,3)}'
ax.plot(mass, fpbh, label=mss)
ax.axhline(1, color="k", ls="--", alpha=0.5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel(r"$f_{pbh}$")
ax.set_xlabel(r"$m_{PBH}\ [M_{\rm sun}]$")
ax.set_xlim(max(mass), min(mass))

plt.legend()
# plt.savefig(PLOTSPATH + "/example_abundances.png")
plt.show()