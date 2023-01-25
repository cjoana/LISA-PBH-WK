"""
Code to evaluate SGWB originated by PBH mergers.  Code provided by Eleni and Satchiko (check?!)
"""

import numpy as np
import scipy.constants as const
import scipy.special as special
from scipy.special import erfc
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.integrate import dblquad
import scipy.optimize as opt


