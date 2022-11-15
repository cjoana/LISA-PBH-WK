import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse

import sys, os
FILEPATH = os.path.realpath(__file__)[:-14]  + "/"
sys.path.append(FILEPATH)
print(FILEPATH)

#Default values, overridden if you pass in command line arguments
listfile_default = FILEPATH + "data_bounds_all.dat" 
outfile_default = FILEPATH + "plots/PBH_bounds.png"
datadir =  FILEPATH + "Data" 

from params_bounds import *


#Load in the filename with the list of bounds and the output filename
parser = argparse.ArgumentParser(description='...')
parser.add_argument('-lf','--listfile', help='File containing list of bounds to include',
                    type=str, default=None)
parser.add_argument('-of','--outfile', help='Filename (with extension) of output plot', 
                    type=str, default=outfile_default)

args = parser.parse_args()
listfile = args.listfile
outfile = args.outfile

sel_files = np.loadtxt(listfile, dtype=str, unpack=True) if listfile else sel_files

print(f"listfile is  {listfile}")
print(f"selected data:   {sel_files}")




plt.figure(figsize=(8,5))

ax = plt.gca()

ax.set_xscale('log')
ax.set_yscale('log')



for bound in sel_files:

    f_bound = bound[0]


    print(f"loading {f_bound}")

    try: 
        x, y = np.loadtxt( datadir + '/' + str(f_bound), unpack=True)
        plt.plot(x,y)
    
    except Exception as e:
        print(f" !!!! dataset {f_bound} has been skipt >> Error:\n {e}")
    
    


#Plotting stuff

# plt.axhspan(1, 1.5, facecolor='grey', alpha=0.5)
   
plt.ylim(1e-9, 1.1)
plt.xlim(1e-18, 1e5)

ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)    
ax.set_xticks(np.logspace(-18, 4, 23), minor=True)
ax.set_xticklabels([], minor=True)
    
plt.xlabel(r'$M_\mathrm{PBH}$ [$M_\odot$]')
plt.ylabel(r'$f_\mathrm{PBH} (M)$')

plt.savefig(outfile, bbox_inches='tight')
    
plt.show()

    