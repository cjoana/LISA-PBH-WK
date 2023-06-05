import math
from munch import Munch

working_parms = Munch(dict())
working_parms.logmmin = -14.0  # Minimum log_10(PBH mass)
working_parms.logmmax = 8.0  # Maximum log_10(PBH mass)
working_parms.Nmass = 1000  # Number of sampling points
working_parms.dlogmass = 0.05

# SGWB from second oder perturbaions
# Set up spectrum limits (kmin, kmax) and number of points nk
# wavenumbers are normalised with respect to the central kp value of the power spectrum
working_parms.kmin = 1e-3
working_parms.kmax = 1e1
working_parms.nk = 2

# parameters for the redshift evolution
working_parms.zmin = 0.
working_parms.zmax = 5.
working_parms.dz = 0.1
working_parms.Nz = 50