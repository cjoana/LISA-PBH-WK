import numpy as np
from params.default_params import *
from user_paths import *

verbose = 1


# Physcal and Cosmological params
physics_units = p_PhysicsUnits()
cosmo_params = p_CosmologicalParameters()

# PowerSpectrum Model
PSModels_params = p_PowerSpectrumModels()
# PS_params = p_PowerSpectrum()
# PS_params.selected_model = 'gaussian'  # e.g.:  'powerlaw' (default), 'broken_powerlaw', 'gaussian', 'lognormal'


# PBH formation params
PBHFormation_params = p_PBHFormation()
# PBHFormation_params.selected_model = 'Musco20'                    # e.g.:  'standard' (default), 'Musco20'
PBHFormation_params.models.Musco20.eta = 0.1

PBHFormation_params.ratio_mPBH_over_mH = 0.2            # Ratio between PBH and Hubble masses at formation
PBHFormation_params.kmsun = 2.1e6                       # frequency related to one Solar Mass 
PBHFormation_params.Pkrescaling = True                  # option to rescale the power spectrum to get a fixed DM fraction
PBHFormation_params.forcedfPBH = 1.                     # Imposed DM fraction made of PBHs
PBHFormation_params.Pkscalingfactor = 1.

# Usage of Thermal History
PBHFormation_params.use_thermal_history = True         # option to include the effect of the equation-of-state changes
                                                  # due to the known thermal history  (needs 'zetacr_file' paths)
PBHFormation_params.data_directory = datadir
PBHFormation_params.zetacr_thermal_file = zetacr_file  # File of the evolution of zeta_cr with thermal history
PBHFormation_params.zetacr_thermal_rad = 1.02          # Reference value of zeta_cr for this file
PBHFormation_params.Gaussian = True                    # Using Gaussian approx. (Currently only option)


# PBH Merging models
MerginRates_params = p_MergingRates()
MerginRates_params.primordial.wanted = True
MerginRates_params.primordial.fsup = 0.0025  # Rate suppression factor from N-body simulations when f_PBH > 0.1
MerginRates_params.clusters.wanted = True
MerginRates_params.clusters.Rclust = 400.





if __name__ == "__main__":

    # This is a simple exemple, call default params, modified as user needs, and do a print check. 

    physics_units = p_PhysicsUnits()
    cosmo_units = p_CosmologicalParameters()
    print("c= ", physics_units.c)

    print("I have chosen a PS : ", PS_params.model)
    print("I have chosen to compute thresholds with  : ", PBHFormation_params.model)

    # This should break the code as variables don't exist (typos)  # Test
    print("Performing error test, trying to set an un-existing parameters...")
    try: 
        # test for typo, `Rclust` exist, `rclust` does not.
        MerginRates_params.clusters.rclust = 400.    
    except AttributeError as E:
        print("Test was succesfull!, error message would have been : ")
        print(f"   >>> AttributeError : {E}")
        print("No worries. ")

