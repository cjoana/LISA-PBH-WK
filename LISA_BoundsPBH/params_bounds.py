import matplotlib as mpl
import argparse


#Specify the plot style
mpl.rcParams.update({'font.size': 10,'font.family':'serif'})
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
#       ["filename",   "legend dset",     "colour" ,  "marker" ,  ...     ]
        ## ["AEDGE.txt",  ],
        ## ["AION100.txt",  ],
        ["CMB.txt",                "CMB",       "purple"     ],
        ["CMB_Kamio_coll.txt",     None,        "purple"     ],
        ["CMB_Kamio_photo.txt",    None,        "purple"     ],
        ["CMB_Serpico_P.txt",      None,        "purple"     ],
        ## ["Cz=0p1.txt",  ],
        ## ["Cz=0p5.txt",  ],
        ["Cz=1.txt",              None,         None  ],
        ["DG.txt",                None,         None ],
        ## ["DLFR.txt",  ],
        ["DynamicalFriction.txt", None,         None ],
        ["EGgamma.txt",          r"$\mathrm{EG}$ $\gamma$",     None ],
        ## ["EGgamma_v2.txt",  ],
        ## ["EPTA.txt",  ],
        ## ["EPTAALL.txt",  ],
        ["EROS.txt",              "EROS",        None   ],
        ["ET.txt",                 "ET",         None   ],
        ["EridanusII.txt",      "Eridanus II",   None   ],
        ["Eros-Macho.txt",      "Eros-MACHO",    None   ], 
        ["FirstClouds.txt",     "First Clouds",  None   ],
        ["GC_ep.txt",           r"GC $e^\pm$",   None   ],
        ["GW.txt",              "GW",            None   ],
        ## ["HSC.txt",  ],
        ["HSC_cons.txt",        "HSC",           "gold"   ],
        ["ICARUS.txt",          "ICARUS",        None   ],
        ## ["INTEGRAL.txt",  ],          #INVISIBLE
        ## ["INTEGRALSPI.txt",  ],       #INVISIBLE
        ## ["Iso-X.txt",  ],             #INVISIBLE
        ## ["JGB.txt",  ],               ## WEIRD
        ["Kepler.txt",          "Kepler",       None],
        ["LEoT1.txt",           "LE oT1",       None],
        ["LEoT2.txt",           "LE oT2",       None],
        ## ["LIGO.txt",  ],
        ## ["LIGO2.txt",  ],
        ["LIGOconstraintMono.txt",      "LIGO",     "brown"],
        ["LalphaForest.txt",            r"L$\alpha$",  "blue"  ],
        ## ["Loebdown.txt",  ],
        ## ["Loebup.txt",  ],
        ## ["LognormalData10.txt",  ],
        ## ["LognormalData7.txt",  ],
        ["MACHO.txt",           "MACHO",     None ],
        ## ["Mono-NoEvo-10.txt",  ],
        ## ["Mono-NoEvo-3-noLV.txt",  ],
        ## ["Mono-NoEvo-3.txt",  ],
        ## ["Mono-NoEvo-7-noLV.txt",  ],
        ## ["Mono-NoEvo-7.txt",  ],
        ## ["Mono-NoEvo.txt",  ],               # SAME AS PlanckDisk
        ## ["NANOGRAV11.txt",  ],
        ## ["NaNoGravMaasF.txt",  ],
        ["OGLE.txt",             "OGLE",     None ],
        ## ["PPTA.txt",  ],
        ## ["PPTAEllis.txt",  ],
        ["PlanckDisk.txt",       "Planck Disk",     'black' ],
        ["PlanckSpherical.txt",  None,             'black' ],
        ## ["Radio.txt",  ],                      # INVISIBLE
        ["SNI.txt",              "SNI",     'green' ],
        ["SNe.txt",              "SNe",     'darkgreen' ],
        ["SegueI.txt",           None,     None ],
        ["UFD.txt",              None,     None ],
        ["V_epm.txt",            r"V $e^\pm$",     "orange" ],
        ## ["WD.txt",  ],                  # Reffuted limit (~ 1e-13)
        ["WideBinaries.txt",     "Wide Binaries",     None ],
        ["XRayB.txt",            "Xray Binaries",     "red" ],
        ["Xray.txt",             None ,     None ],
        ## ["delta_c_EOS.txt",  ],   ##??? 
        ## ["gstar(T).txt",  ],          ## 4 columns  with weird first values (1e-6 -- 1e16)
        ## ["mudistorsion.txt",  ],      #INVISIBLE
        ## ["np.txt",  ],     ## ??
        ## ["text.txt",  ],   ## ??
#
]