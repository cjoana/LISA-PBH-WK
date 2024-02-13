import sys, os
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))

PROJECTPATH = ROOTPATH
SOURCEPATH = os.path.abspath(os.path.join(PROJECTPATH, 'source'))
PARAMSPATH = os.path.abspath(os.path.join(PROJECTPATH, 'params'))
DATAPATH =  os.path.abspath(os.path.join(PROJECTPATH, 'data'))
PLOTSPATH = os.path.abspath(os.path.join(PROJECTPATH, 'plots'))

prjdir = PROJECTPATH
datadir = DATAPATH
plotsdir = PLOTSPATH
zetacr_file = datadir + "/zetacr.dat"




