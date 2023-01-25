import numpy as np
from munch import Munch
from functions import *
from user_paths import prjdir, datadir, zetacr_file, figdir

"""
This file contain the following dictionaries:
    *   physics_units   (c, mpc, AU, kb, etc)
    *   cosmo_params    (H0, ns, etc. )

It contains Pk models as dictionaries:     e.g. Pk_models['mymodel'] = {"my_parms": 0, ...}
    *   Pk_model.powerlaw
    *   Pk_model.lognormal
    *   Pk_model.gaussian
    *   Pk_model.broken_powerlaw

It contains params of PBH formation models
    *   standard formation model
    *   Musco20 model

It contains params of Merging rates models
    *   PBH binareis
    *   Clusters
"""

#######################################################################
# Units and cosmological parameters
#######################################################################

# units and constants
physics_units = Munch(dict())
physics_units['c'] = 2.997924e8  # (*Speed of light, [m/s] *)
physics_units['mpc'] = 3.085678e22  # (* Megaparsec [m] *)
physics_units['pc'] = 3.086e16  # (*parsec [m]*)
physics_units['G'] = 6.67428e-11  # (* Gravitational constant [m^3/kg/s^2] *)
physics_units['m_sun'] = 1.989e30  # (* Sun mass [kg]*)
physics_units['year'] = 365.25 * 24 * 3600  # (* year in seconds[s] *)
physics_units['hbar'] = 6.62607e-34 / (2. * np.pi)  # (* reduced planck constant in m^2 kg /s*)
physics_units['hp'] = 6.582e-16  # (* Planck constant in eV s *)
physics_units['AU'] = 1.4960e11  # (* Astronomical unit [m]*)
physics_units['kb'] = 8.617343e-5  # (* Boltzmann constant [eV /K] *);
physics_units['eV_in_J'] = 1.60217e-19  # (*eV in Joule *)
physics_units['l_planck'] = 1.61e-35  # (*Planck length [m]*)
physics_units['rho_planck'] = 5.155e96  # (*Planck energy [kg/m^3]*)
physics_units['m_planck'] = 1.22e19  # (*Planck mass [eV]*)

# Standard cosmological parameters
cosmo_params = Munch(dict())
cosmo_params['ns'] = 0.961
cosmo_params['As'] = 2.1e-9
cosmo_params['Omb'] = 0.0456
cosmo_params['Omc'] = 0.245
cosmo_params['h'] = 0.7
cosmo_params['Nur'] = 3.046
cosmo_params['TCMB'] = 2.726
cosmo_params['kstar'] = 0.05  # TODO: check this number (is kp?)
cosmo_params['kp'] = 2.e6
# derived cosmological parameters
cosmo_params['H0'] = cosmo_params.h * 100000. / physics_units.mpc  # (* Hubble rate today, [s^-1] *)
cosmo_params['rhocr'] = 3. * cosmo_params.H0 ** 2 / (8. * np.pi * physics_units.G)  # (* Critical Density [kg/m^3] *)
cosmo_params['ar'] = 7.5657e-16  # (* Stephan's constant in J m^-3 K^-4 *);
cosmo_params['Omr'] = cosmo_params.ar * cosmo_params.TCMB ** 4 / cosmo_params.rhocr / physics_units.c ** 2  # (* Density of photons *)
cosmo_params['Omnu'] = cosmo_params.Omr * 7. / 8 * cosmo_params.Nur * (4. / 11.) ** (4. / 3.)  # (* Density of neutrinos *)
cosmo_params['OmLambda'] = 1. - (cosmo_params.Omc + cosmo_params.Omb + cosmo_params.Omr + cosmo_params.Omnu)  # (* Density of Dark Energy *)

#######################################################################
# Curvature power sperctrum Models:   Pk_models
#######################################################################

Pk_models = Munch(dict())
Pk_models.kp = 2.e6
Pk_models.Pk_model = 'default'
Pk_models.default = Munch(dict())
Pk_models.default.Pk_model = 'powerlaw'

#  Power law:
Pk_models.powerlaw = Munch(dict())
Pk_models.powerlaw.Pk_model = "powerlaw"
Pk_models.powerlaw.ktrans = 1.e0      # Scale of the transition between CMB amplitude and PBH amplitude
Pk_models.powerlaw.nsPBH = 0.97       # Spectral index
Pk_models.powerlaw.kp = 2.e6          # Reference scale [mpc^-1] (2.e6 corresponds to mPBH = 1 Msun)
Pk_models.powerlaw.AsPBH = 0.0205     # Power spectrum amplitude at the reference scale kp.

# "Log-normal":
Pk_models.lognormal = Munch(dict())
Pk_models.lognormal.Pk_model = "lognormal"
Pk_models.lognormal.kp = 2.e6         # Reference scale [mpc^-1] (2.e6 corresponds to mPBH = 1 Msun)
Pk_models.lognormal.AsPBH = 0.0205    # Power spectrum amplitude at the reference scale kp.
Pk_models.lognormal.sigma = 1.        # Power spectrum amplitude at the reference scale kp.

# "Gaussian":
Pk_models.gaussian = Munch(dict())
Pk_models.gaussian.Pk_model = "gaussian"
Pk_models.gaussian.kp = 2.e6           # Reference scale [mpc^-1] (2.e6 corresponds to mPBH = 1 Msun)
Pk_models.gaussian.AsPBH = 0.0205      # Power spectrum amplitude at the reference scale kp.
Pk_models.gaussian.sigma =  0.2        # Power spectrum variance.

# "Broken power-law":
Pk_models.broken_powerlaw = Munch(dict())
Pk_models.broken_powerlaw.Pk_model = "broken_powerlaw"
Pk_models.broken_powerlaw.kp = 2.e6      # Reference scale [mpc^-1] (2.e6 corresponds to mPBH = 1 Msun)
Pk_models.broken_powerlaw.AsPBH = 0.0205 # Power spectrum amplitude at the reference scale kp.
Pk_models.broken_powerlaw.kc = 2.e6      # k critical: splits btw low/high
Pk_models.broken_powerlaw.kp_low = Pk_models.broken_powerlaw.kp
Pk_models.broken_powerlaw.kp_high = Pk_models.broken_powerlaw.kp
Pk_models.broken_powerlaw.AsPBH_low = Pk_models.broken_powerlaw.AsPBH
Pk_models.broken_powerlaw.AsPBH_high = Pk_models.broken_powerlaw.AsPBH
Pk_models.broken_powerlaw.ns_low = -0.1
Pk_models.broken_powerlaw.ns_high = -0.5

# "Power spectrum from multifield"
Pk_models.multifield = Munch(dict())
Pk_models.multifield.Pk_model = "multifield"
Pk_models.multifield.kf = 2e6          # scale H-crossing at sharp turn  [mpc^-1]
Pk_models.multifield.P0 = 2e-9        # Amplitude Pk in absence of transient instability
Pk_models.multifield.eta = 4.           # TODO: check value
Pk_models.multifield.delta = 2.2          # TODO: check value
Pk_models.multifield.kp = Pk_models.multifield.kf


# "Power spectrum from axion-gauge"
Pk_models.axion_gauge = Munch(dict())
Pk_models.axion_gauge.Pk_model = "axion_gauge"
Pk_models.axion_gauge.kp = 2.1e6           # Reference scale [mpc^-1] (2.e6 corresponds to mPBH = 1 Msun)
Pk_models.axion_gauge.As_vac = 0.0205      # TODO: k-dependent
Pk_models.axion_gauge.AsPBH = 0.0205      # TODO: k-dependent
Pk_models.axion_gauge.sigma = 1.          # TODO: k-dependent

# "Power spectrum from preheating"
Pk_models.preheating = Munch(dict())
Pk_models.preheating.Pk_model = "preheating"
Pk_models.preheating.kend = 2.e6     #
Pk_models.preheating.Hstar = 1.0e14      # TODO: k-dependent
Pk_models.preheating.e1 = 0.01           # TODO: k-dependent
Pk_models.preheating.e2 = 0.0001            # TODO: k-dependent
Pk_models.preheating.C = -0.7296          
Pk_models.preheating.kp = Pk_models.preheating.kend


# "Poser spectrum set up by user"
Pk_models.user_import = Munch(dict())
Pk_models.user_import.Pk_model = "user_import"

# "Power from a file provided by the user
# TBC with an example of file


#######################################################################
#  PBH formation Model
#######################################################################
#TODO: Reduce redundancy of params for global/particular models

PBHForm = Munch(dict())
PBHForm.ratio_mPBH_over_mH = 0.8  # Ratio between PBH and Hubble masses at formation
PBHForm.kmsun = 2.1e6
PBHForm.Pkrescaling = True  # option to rescale the power spectrum to get a fixed DM fraction
PBHForm.forcedfPBH = 1.  # Imposed DM fraction made of PBHs
PBHForm.Pkscalingfactor = 1.
PBHForm.use_thermal_history = True  # option to include the effect of the equation-of-state changes due to the known thermal history
PBHForm.data_directory = datadir
PBHForm.zetacr_thermal_file = zetacr_file  # File of the evolution of zeta_cr with thermal history
PBHForm.zetacr_thermal_rad = 1.02  # Reference value of zeta_cr for this file
PBHForm.Gaussian = True
PBHForm.PBHform_model = 'default'


# intro models
PBHForm.models = Munch(dict())
PBHForm.models.default = Munch(dict())
PBHForm.models.default.PBHform_model = 'standard'

# standard formation model
PBHForm.models.standard = Munch(dict())
PBHForm.models.standard.PBHform_model = "standard"
PBHForm.models.standard.deltacr_rad = 0.41  #1.02

# Musco20 formation model
PBHForm.models.Musco20 = Munch(dict())
PBHForm.models.Musco20.PBHform_model = "Musco20"
PBHForm.models.Musco20.eta = 0.1
PBHForm.models.Musco20.k_star = PBHForm.kmsun   # TODO: Check


#######################################################################
# Model merging rates
#######################################################################
MergingRates_models = Munch(dict())

# Primordial binaries
MergingRates_models.primordial = Munch(dict())
MergingRates_models.primordial.wanted = True
MergingRates_models.primordial.norm = 1.6e6
MergingRates_models.primordial.fsup = 0.0025  # Rate suppression factor from N-body simulations when f_PBH > 0.1

# Tidal capture in clusters
MergingRates_models.clusters = Munch(dict())
MergingRates_models.clusters.wanted = True
MergingRates_models.clusters.Rclust = 400.








if __name__ == "__main__":
    # Munch allows to set-up/extract values as  both dictionary and class ways
    print("c= ", physics_units.c, physics_units['c'])
