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
# outfile_default = FILEPATH + "plots/PBH_bounds.png"
datadir =  FILEPATH + "data" 

outfile = FILEPATH + "plots/PBH_bounds.png"
outfile2 = FILEPATH + "plots/PBH_bounds_summary.png"

from params_bounds import *


#Load in the filename with the list of bounds and the output filename
parser = argparse.ArgumentParser(description='...')
parser.add_argument('-lf','--listfile', help='File containing list of bounds to include',
                    type=str, default=None)
# parser.add_argument('-of','--outfile', help='Filename (with extension) of output plot', 
#                     type=str, default=outfile_default)

args = parser.parse_args()
listfile = args.listfile
# outfile = args.outfile

sel_files = np.loadtxt(listfile, dtype=str, unpack=True) if listfile else sel_files

print(f"listfile is  {listfile}")
print(f"selected data:   {sel_files}")


fig, ax = plt.subplots(1,1, figsize=(8,5))

fig2, ax2 = plt.subplots(1,1, figsize=(8,5))


for bound in sel_files:

    f_bound = bound[0]
    lbl = bound[1]
    color = bound[2]

    errs = []
    print(f"loading {f_bound}")

    try: 
        x, y = np.loadtxt( datadir + '/' + str(f_bound), unpack=True)

        # lbl = str(f_bound[:-4])
        # if color: 
        #     plt.plot(x,y,  label=lbl, color=color)
        # else:
        #     plt.plot(x,y,  label=lbl)

        asort = np.argsort(x)
        x,y = [x[asort], y[asort] ] 
        ax.plot(x,y,  label=lbl, color=color)
        ax2.fill_between(x, y, y2=1, color="gray", interpolate=True) 
    
    except Exception as e:
        mess = f" !!!! dataset {f_bound} has been skipt >> Error:\n {e}"
        errs.append[mess]
        print(mess)
    

for e in errs:
    print(e)



#Plotting stuff

# plt.axhspan(1, 1.5, facecolor='grey', alpha=0.5)

ax.legend(ncol=2)

for axs in [ax, ax2]:   
    axs.set_ylim(1e-10, 1.05)
    axs.set_xlim(5e-19, 1e8)

    axs.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)    
    axs.set_xticks(np.logspace(-18, 4, 23), minor=True)
    axs.set_xticklabels([], minor=True)
        
    axs.set_xlabel(r'$M_\mathrm{PBH}$ [$M_\odot$]')
    axs.set_ylabel(r'$f_\mathrm{PBH} (M)$')

    axs.set_xscale('log')
    axs.set_yscale('log')



fig.savefig(outfile, bbox_inches='tight', dpi=1200)
fig2.savefig(outfile2, bbox_inches='tight', dpi=1200)
    
# fig.show()

    
