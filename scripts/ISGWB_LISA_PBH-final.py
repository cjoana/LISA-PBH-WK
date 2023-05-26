import scipy
import math
import numpy as np
from scipy.integrate import dblquad

# Set up spectrum limits (kmin, kmax) and number of points nk
# wavenumbers are normalised with respect to the central ks of the power spectrum
kmin = 1e-3
kmax = 1e1
nk = 50

# peak power spectrum wavenumber, k = 2 pi frequency, set is the peak sensitivity frequency of LISA as an example.
ks = 2*math.pi*3.4*1e-3 #Hz

# Lognormal power spectrum width and aplitude
sigmaps=1/2
Aps = 0.044


# Lognormal curvature perturbation power spectrum
def PS(k):
    return Aps/(2 *math.pi * sigmaps**2)**0.5 * np.exp(- 1/(2*sigmaps**2)*np.log(k)**2)



# Integrated transfer functions
def IC2(d,s):
    return -36* math.pi * (s**2 +d**2-2)**2/(s**2-d**2)**3 *np.heaviside(s-1, 1)

def IS2(d,s):
    return -36* (s**2 +d**2-2)/(s**2-d**2)**2 *((s**2 +d**2-2)/(s**2-d**2) * np.log((1-d**2)/np.absolute(s**2-1))+2)
def IcsEnvXY(x,y):
    return (IC2(np.absolute(x-y)/(3**0.5),np.absolute(x+y)/(3**0.5))**2 + IS2(np.absolute(x-y)/(3**0.5),np.absolute(x+y)/(3**0.5))**2)**0.5

# Integral returning the spectrum
def compint(kvval, sigmaps):
    value, error = dblquad(lambda x, y: 
        x**2/y**2 * (1-(1+x**2-y**2)**2/(4* x**2))**2
        * PS(kvval*x)
        * PS(kvval*y)
        * IcsEnvXY(x,y)**2
        ,
        10**(- 4*sigmaps)/kvval,10**(4*sigmaps)/kvval, lambda x: np.absolute(1-x), lambda x: 1+x)
    return value

# compute:
kvals = np.logspace(np.log10(kmin),np.log10(kmax), nk)

kres =np.array([compint(xi,sigmaps) for xi in kvals])

# coefficient due to thermal history see Eq. (2.11) https://arxiv.org/pdf/1810.12224.pdf
# to be updated depending on the reference peak of the spectrum, to integrated with the rest of the code
cg = 0.4
Omega_r_0 = 2.473*1e-5
norm = cg*Omega_r_0/972.


# final spectrum
#  X
print(ks*kvals)
#  Y
print(norm*kres)

# show plot
import matplotlib.pyplot as plt
plt.plot(ks*kvals,norm*kres) # Create line plot with yvals against xvals
plt.yscale('log')
plt.xscale('log')
# plt.ylim(1e-5, 1e4)
# plt.plot(x, y, label='First Line')
# plt.plot(x2, y2, label='Second Line')
plt.xlabel('Wavenumber $k$')
plt.ylabel('$\Omega_{GW}$')
#plt.title('Interesting Graph\nCheck it out')
# plt.legend()
plt.show() # Show the figure
