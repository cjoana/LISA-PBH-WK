import numpy as np
import primbholes 
from power_spectrum import PowerSpectrum

# own definition 
def my_PS(k):
    ...

# call primbholes
pb = primbholes(ps_function = my_PS)

# compute signature
fpbh = lambda mass: pb.get_fPBH(mass)
beta = lambda mass: pb.get_beta(mass)

rates_EB = lambda masses: pb.get_rates_primordial(masses) 
rates_LB = lambda masses: pb.get_rates_clusters(masses) 

GWB_EB = lambda freq: pb.Get_GW_bkg_primordial_binary(freq)
GWB_LB = lambda freq: pb.Get_GW_bkg_cluster_binary(freq)  