import numpy as np
from scipy import interpolate
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import math
import cmath
from matplotlib import rc

#Plotting Configurations
f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
fmt = mticker.FuncFormatter(g)
rc('text', usetex=True)

#Fixed Physical Parameters
m_Pl=1.22*10**(19.) #non-reduced Planck mass in GeV
M_Pl=m_Pl/np.sqrt(8.*np.pi) #reduced Planck Mass in GeV
gamma = 0.2 # The gravitational collapse efficiency parameter
M_sun = 10.**(57.) #in GeV

#Defining the curvature power spectrum at the end of the preheating instability
def P_zeta(rho_inf,rho_gamma):
	#Consistency Checks
	if rho_gamma>rho_inf:
		print 'Wrong parameters inserted, rho_inf or rho_gamma. Attention: rho_inf>rho_gamma. '

	elif rho_gamma/M_Pl**(4.)<(4./(125.*np.sqrt(3.)*np.pi**(5.)))*(rho_inf/M_Pl**(4.))**(5./2.):
		print 'Too many PBHs produced, Omega_PBH>1 at the end of the preheating instabiliy.'

	else:
		H_end = np.sqrt(rho_inf/(3.*M_Pl**(2.)))
		phi_end_n = 1.00705*M_Pl #It's constant independent of the value of mass m for V= m^2*phi^2/2
		m = 2.*H_end*M_Pl/phi_end_n

		#Boundary Conditions
		N=100000
		phi_in = 20.*M_Pl
		H_in = np.sqrt(((m**(2))*phi_in**(2))/(6*M_Pl**(2)))
		phi_dot_in = -(m**(2))*phi_in/(3*H_in**(2))

		#The Klein Gordon and Friedmann Equations combined in terms of e-folds
		def KG(U, t):
			x = U[0] #The field phi
			y = U[1] #Fist order time derivative of the field
			dphi_dN = y
			dphi_dN_dN = -(3. - y**(2)/(2*M_Pl**(2)))*(y + 2*(M_Pl**(2))/x)
			return [dphi_dN,dphi_dN_dN]

		#Solving the combined Klein Gordon and Friedmann Equations
		U0 = [phi_in, phi_dot_in]
		N_in=0
		N_fin=101
		N_range = np.linspace(N_in, N_fin, N)
		Us = odeint(KG, U0, N_range)
		phi = Us[:,0]
		phi_dot= Us[:,1]

		#Interpolating the phi and phi_dot fields
		Phi = interpolate.interp1d(N_range,phi)
		Phi_dot = interpolate.interp1d(N_range,phi_dot)

		#The scale factor
		def sf(x):
			return np.exp(x)

		#The Hubble Parameter
		def H(x):
			return np.sqrt(((m**(2))*Phi(x)**(2))/(6.*(M_Pl**(2))*(1.-Phi_dot(x)**(2)/(6.*M_Pl**(2)))))

		#The first slow-roll parameter epsilon_1
		def epsilon1(x):
			return Phi_dot(x)**(2)/(2*(M_Pl**(2)))

		#The second slow-roll parameter epsilon_2
		def epsilon2(x):
			return 6.*((epsilon1(x)/3.) - ((m**(2.))*Phi(x)/(3.*(H(x)**(2))*Phi_dot(x)))-1.)

		#Finding the end of inflation - N_end and k_end = a_end*H_end
		a = N_range[0]
		b = N_range[len(N_range)-1]
		while(epsilon1(b)-epsilon1(a)>10**(-2)):
			c=(a+b)/2.
			if epsilon1(c)-1.>0:
				b=c
			else:
				a=c
		N_end = b
		a_end = sf(N_end)
		k_end = a_end*H_end #The comoving scale exiting the Hubble radius at the end of inflation

		#Finding the Horizon crossing time for a mode k
		def N_star(k):
			a = N_range[0]
			b = N_end
			while(b-a>10**(-2)):
				c=(a+b)/2
				if sf(c)*H(c)-k>0.:
					b=c
				else:
					a=c
			return b

		# Defining the range of modes entering the preheating instability from above k\in [k_min,k_end].
		# x is defined as x=k/k_end (auxiliary variable)
		x_min = (rho_gamma/rho_inf)**(1./6.)
		x_range = np.linspace(np.log(x_min),np.log(1.),1000)
		x_range = np.exp(x_range)

		#Constructing H_star(x), epsilon1_star(x) and epsilon2_star(x)
		H_star_array = [H(N_star(k_end*x_i)) for x_i in x_range]
		epsilon1_star_array = [epsilon1(N_star(k_end*x_i)) for x_i in x_range]
		epsilon2_star_array = [epsilon2(N_star(k_end*x_i)) for x_i in x_range]

		H_star = interpolate.interp1d(x_range,H_star_array)
		epsilon1_star = interpolate.interp1d(x_range,epsilon1_star_array)
		epsilon2_star = interpolate.interp1d(x_range,epsilon2_star_array)

		#The curvature_power_spectrum at the end of the preheating instability
		def curvature_power_spectrum(x):
			C = -0.7296
			k = x*k_end
			curv_PS = ((H_star(x)**(2))/(8*(np.pi**(2))*(M_Pl**(2))*epsilon1_star(x)))*(1+(x)**(2))*(1-2*(C+1)*epsilon1_star(x)-C*epsilon2_star(x))
			return curv_PS
		curv_PS_range = [curvature_power_spectrum(x) for x in x_range]

		#The density_power_spectrum at the end of the preheating instability
		dens_PS_range = x_range**(4.)*((rho_inf/rho_gamma)**(2./3.))*((6./5.)**(2))*curv_PS_range
		return dens_PS_range

#The PBH mass in solar masses
def mass(x,rho_inf,rho_gamma):
	x_min = (rho_gamma/rho_inf)**(1./6.)
	x_max = 1.
	if x_min<x<x_max:
		return gamma*(((3.*(M_Pl**(2)))**(3./2.))/np.sqrt(rho_inf))*(x**(-3.))/M_sun
	else:
		print 'Wrong parameters inserted, x, rho_inf or rho_gamma.'


#Specifying rho_end and rho_reh
rho_inf = (10.**(14))**(4.) #in GeV^4
rho_gamma = (10.**(7))**(4.) #in GeV^4

#Find the relevant range of modes in terms of the variable x=k/k_end.
x_min = (rho_gamma/rho_inf)**(1./6.)
x_range = np.linspace(np.log(x_min),np.log(1.),1000)
x_range = np.exp(x_range)

#Plotting the curvature power spectrum at the end of the prehating instability
plt.plot(x_range,P_zeta(rho_inf,rho_gamma), label=r'$\rho^{1/4}_\mathrm{inf}$' + "={}".format(fmt(rho_inf**(1./4.))) + r'$\mathrm{GeV}$, $\rho^{1/4}_\mathrm{\Gamma}$' + "={}".format(fmt(rho_gamma**(1./4.))) + r'$\mathrm{GeV}$')
plt.xlabel(r'$k/k_\mathrm{end}$')
plt.ylabel(r'$\mathcal{P}_\mathrm{\delta}(t_\mathrm{\Gamma},k/k_\mathrm{end})$')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='best')
plt.show()
