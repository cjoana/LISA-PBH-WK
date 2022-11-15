import numpy as np
import scipy.constants as const
import scipy.special as special
# from scipy.special import erfc
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.integrate import dblquad
import scipy.optimize as opt
import os
import math
import matplotlib as mp
import matplotlib.pyplot as plt
# import classy
from numpy import ma
from matplotlib import ticker, cm
from munch import Munch

from default_params import cosmo_params, physics_units


class Cosmology:

    def __init__(self):
        cp = cosmo_params
        self.ns = cp.ns
        self.As = cp.As
        self.Omb = cp.Omb
        self.Omc = cp.Omc
        self.h = cp.h
        self.Nur = cp.Nur
        self.TCMB = cp.TCMB
        self.kstar = cp.kstar
        self.H0 = cp.H0
        self.rhocr = cp.rhocr
        self.ar = cp.ar
        self.Omr = cp.Omr
        self.Omnu = cp.Omnu
        self.OmLambda = cp.OmLambda

    def rhoCDM(self, a):
        # Returns the CDM density at scale factor a
        rhoCDM = self.Omc * self.rhocr / a ** 3
        return rhoCDM

    def rhob(self, a):
        # Returns the baryon density at scale factor a
        rho = self.Omb * self.rhocr / a ** 3
        return rho #rhoCDM

    def H(self, a):
        # Returns the Hubble rate parameter (in s^-1) at scale factor a
        H = self.H0 * np.sqrt((self.Omb + self.Omc) / a ** 3 + (self.Omr + self.Omnu) / a ** 4 + self.OmLambda)
        return H

    def Hmass(self, a):
        pu = physics_units
        # Returns the mass in the Hubble radius
        Hmass = (3. * self.H(a) ** 2 / (8. * np.pi * pu.G)) / (self.H(a) / pu.c) ** 3
        return Hmass


class CurvFluctuations:

    def __init__(self):
        self.type = 'Gaussian'


class PBHFormation:

    def __init__(self):
        self.type = 'Gaussian'


class MergingRates:

    def __init__(self):
        self.type = 'Gaussian'


class PkSpectrum:

    def __init__(self):
        self.type = 'Gaussian'


##############################3

if __name__ == "__main__":

    mycosmo = Cosmology()

    H = mycosmo.H(2)
    print(H)