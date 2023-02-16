import matplotlib as mpl
import argparse


#Specify the plot style
mpl.rcParams.update({'font.size': 12,'font.family':'serif'})
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.minor.width'] = 1
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rc('text', usetex=True)

mpl.rcParams['legend.edgecolor'] = 'inherit'


sel_files = [
#       ["filename",   "legend dset",     "colour" , "order",  "marker" ,  ...     ]
        #Low Mass
        ["V_epm.txt",            r"V $e^\pm$",     "r" , 1 ],
        ["EGgamma.txt",          r"$\mathrm{EG} \gamma$",     "g" , 1 ],
        ["GC_ep.txt",           r"GC $e^\pm$",     "b", 1 ],
        #
        # Mid Mass
        ["HSC_cons.txt",        "HSC",           "b"   , 1 ],
        ["Kepler.txt",          "Kepler",       'gold' , 2 ],
        ["EROS.txt",              "EROS",        "black"   , (1,1,2) ],
        ["OGLE.txt",             "OGLE",     'gray' , 1 ],
        #
        # High Mass
        ["ICARUS.txt",          "ICARUS",        'm'   , 1 ],
        ["SNe.txt",              "SNe",     'orange' , 2 ],
        ["GW.txt",              "GW (LVK)",            'brown'   , 2 ],
        ["MACHO.txt",           "MACHO",     'purple' , 2 ],
        ["PlanckSpherical.txt",  "Planck Spherical",   'gray' , 2 ],
        ["PlanckDisk.txt",       "Planck Disk",     'black' , 1 ],
        ["CMB.txt",                "CMB",       "purple"  , 3 ],
        ["CMB_Kamio_coll.txt",     "CMB K1",        "r"   , 3 ],
        ["CMB_Kamio_photo.txt",    "CMB K2",        "b"     , 3 ],
        ["CMB_Serpico_P.txt",      "CMB S",        "k"     , 3 ],
        ["SegueI.txt",           "segueiI" ,  "green",  2 ],
        ["EridanusII.txt",      "Eridanus II",   'm'   , 2 ],
        ["xray_ziparo22.txt",    "Xray ziparo22" ,     'orange' , 3 ],
        ["XRayB.txt",            "Xray Binaries",     "red" , 3 ],
        ["LalphaForest.txt",            r"L $\alpha$",  "c"  , 2 ],
        ["DynamicalFriction.txt", "Dyn. friction",  'darkgreen' , 1 ],
        ["FirstClouds.txt",     "First Clouds",  'm'   , 2 ],
        ["LEoT1.txt",           "LE oT1",       'r',    2 ],
        ["UFD.txt",              "UFD" ,        'g',     3 ],
        ["WideBinaries.txt",     "Wide Binaries",   'm' , 3 ],
        ["cmb_dist.txt",           "CMB dist",  "blue"     , 1 ],
        ["DG.txt",                "DG",         'brown' , 1 ],
        #
        #
        #
        #
        ## ["LEoT2.txt",           "LE oT2",       'k',    None ],
        ## ["Xray.txt",             "Xray2" ,     "green" , 2 ],
        ## ["LIGOconstraintMono.txt",      "LIGO",     "brown", 3 ],  ## Same as GW
        ## ["SNI.txt",              "SNI",     'orange' , 3 ],  # Samw as SNe
        ## ["AEDGE.txt", None , None ],
        ## ["AION100.txt", None  , None ],
        ## ["Cz=1.txt",              "Cz=1",         'r'  , None ],   #Same as ICARUS
        ## ["Cz=0p1.txt", None  , None ],
        ## ["Cz=0p5.txt", None  , None ],
        ## ["Eros-Macho.txt",      "Eros-MACHO",    'c'   , None ],   #Same as EROS
        ## ["EGgamma_v2.txt",  , None ],
        ## ["EPTA.txt",  , None ],
        ## ["EPTAALL.txt",  , None ],
        ## ["DLFR.txt",  , None ],
        # ["ET.txt",                 "ET",         None   , None ],
        ## ["HSC.txt",  , None ],
        ## ["INTEGRAL.txt",  , None ],          #INVISIBLE
        ## ["INTEGRALSPI.txt",  , None ],       #INVISIBLE
        ## ["Iso-X.txt",  , None ],             #INVISIBLE
        ## ["JGB.txt",  , None ],               ## WEIRD
        ## ["LIGO.txt",  , None ],
        ## ["LIGO2.txt",  , None ],
        ## ["Loebdown.txt",  , None ],
        ## ["Loebup.txt",  , None ],
        ## ["LognormalData10.txt",  , None ],
        ## ["LognormalData7.txt",  , None ],
        ## ["Mono-NoEvo-10.txt",  , None ],
        ## ["Mono-NoEvo-3-noLV.txt",  , None ],
        ## ["Mono-NoEvo-3.txt",  , None ],
        ## ["Mono-NoEvo-7-noLV.txt",  , None ],
        ## ["Mono-NoEvo-7.txt",  , None ],
        ## ["Mono-NoEvo.txt",  , None ],               # SAME AS PlanckDisk
        ## ["NANOGRAV11.txt",  , None ],
        ## ["NaNoGravMaasF.txt",  , None ],
        ## ["PPTA.txt",  , None ],
        ## ["PPTAEllis.txt",  , None ],
        ## ["Radio.txt",  , None ],                      # INVISIBLE
        ## ["WD.txt",  , None ],                  # Reffuted limit (~ 1e-13)
        ## ["delta_c_EOS.txt",  , None ],   ##??? 
        ## ["gstar(T).txt",  , None ],          ## 4 columns  with weird first values (1e-6 -- 1e16)
        ## ["mudistorsion.txt",  , None ],      #INVISIBLE
        ## ["np.txt",  , None ],     ## ??
        ## ["text.txt",  , None ],   ## ??
#
]