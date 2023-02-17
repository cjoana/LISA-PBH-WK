"""
Code to evaluate merger rates given a mass distribution f(m1, m2, z), with PBH masses m1, m2 at a redshift z. 
Basic models are included and others can be expanded easily. 
"""

import numpy as np
import scipy.constants as const
import scipy.special as special
from scipy.special import erfc
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.integrate import dblquad
import scipy.optimize as opt


class MergerRates(): 

    def __init__(self): 
        #TODO : #set up everyting!

        self.Rclust = None
        self.Nmass = None
        self.mPBHtable = None
        self.logfofmPBHtable = None

        self.fsup = 1

        # output
        self.rate_prim = None
        self.rate_clust = None

        
        pass

        #### 
    def rates_primordial(self):
        # Computes the merging rates of primordial binaries
        norm = 1.6e6
        rates = np.zeros((self.Nmass, self.Nmass))
        for ii in range(self.Nmass):
            m1 = self.mPBHtable[ii]
            for jj in range(ii):
                m2 = self.mPBHtable[jj]
                rates[ii, jj] = norm * self.fsup * 10. ** self.logfofmPBHtable[ii] * 10. ** self.logfofmPBHtable[jj] * \
                    (m1 + m2) ** (-32. / 37.) * (m1 * m2 / (m1 + m2) ** 2) ** (-34. / 37.)
        return rates

    def rates_clusters(self):
        # Computes the merging rates for tidal capture in PBH clusters
        norm = self.Rclust
        rates = np.zeros((self.Nmass, self.Nmass))
        for ii in range(self.Nmass):
            m1 = self.mPBHtable[ii]
            for jj in range(ii):
                m2 = self.mPBHtable[jj]
                rates[ii, jj] = norm * 10. ** self.logfofmPBHtable[ii] * 10. ** self.logfofmPBHtable[jj] * \
                    (m1 + m2) ** (10. / 7.) / (m1 * m2) ** (5. / 7.)
        return rates

    def eval_oldcode(self):
        # Compute the merging rates
        print("Step 3:  Computation of the PBH merging rates")
        print("====")
        self.rate_prim = np.zeros((self.Nmass, self.Nmass))
        self.rate_clust = np.zeros((self.Nmass, self.Nmass))
        if self.merging_want_primordial == True:
            self.rate_prim = self.rates_primordial()
            print("Merging rates of primordial binaries have been calculated")
            print("====")

        if self.merging_want_clusters == True:
            self.rate_clust = self.rates_clusters()
            print("Merging rates of binaries formed by capture in clusters have been calculated")
            print("====")

        print("End of code, at the moment...  Thank you for having used PrimBholes")