import numpy as np
from munch import Munch
from params.__default_params import *
from user_paths import *

verbose = 1

# Pk Model
Pk_models.Pk_model = 'powerlaw'  # e.g.:  'powerlaw' (default), 'broken_powerlaw', 'gaussian', 'lognormal'


# PBH formation params
PBHForm.PBHform_model = 'Musco20'           # e.g.:  'standard' (default), 'Musco20'
PBHForm.models.Musco20.eta = 0.1

PBHForm.ratio_mPBH_over_mH = 0.2            # Ratio between PBH and Hubble masses at formation
PBHForm.kmsun = 2.1e6
PBHForm.Pkrescaling = True                  # option to rescale the power spectrum to get a fixed DM fraction
PBHForm.forcedfPBH = 1.                     # Imposed DM fraction made of PBHs
PBHForm.Pkscalingfactor = 1.

# Usage of Thermal History
PBHForm.use_thermal_history = True          # option to include the effect of the equation-of-state changes
                                            # due to the known thermal history  (needs 'zetacr_file' paths)
PBHForm.data_directory = datadir
PBHForm.zetacr_thermal_file = zetacr_file  # File of the evolution of zeta_cr with thermal history
PBHForm.zetacr_thermal_rad = 1.02          # Reference value of zeta_cr for this file
PBHForm.Gaussian = True                    # Using Gaussian approx. (Currently only option)


# PBH Merging models
MergingRates_models.primordial.wanted = True
MergingRates_models.primordial.fsup = 0.0025  # Rate suppression factor from N-body simulations when f_PBH > 0.1
MergingRates_models.clusters.wanted = True
MergingRates_models.clusters.Rclust = 400.