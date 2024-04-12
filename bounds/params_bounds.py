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
        #["OGLE.txt",             "OGLE",     'gray' , 1 ],
        ["OGLE2024.txt",         "OGLE",     'gray' , 1 ],				# arxiv: 2403.02386
        #
        # High Mass
        ["ICARUS.txt",          "ICARUS",        'm'   , 1 ],
        ["SNe.txt",              "SNe",     'orange' , 2 ],
        ["xray_ziparo22.txt",    "XRay bkg" ,     'orange' , 1 ],       #  arxiv: 2209.09907
        ["PlanckSpherical.txt",  "CMB SPIK2",   'gray' , 2 ],			 # 	arxiv: 2002.10771 (2 sph)
        ["PlanckDisk.txt",       "CMB SPIK1",     'black' , 1 ],		 # 	arxiv: 2002.10771 (1 disk)
        ["GW.txt",               "GW (LVK)",            'brown'   , 2 ],
        ["CMB_Kamio_coll.txt",   "CMB AK1",        "r"   , 3 ],  		 #  arxiv: 1612.05644 (1)
        ["CMB_Kamio_photo.txt",  "CMB AK2",        "b"     , 3 ],      #  arxiv: 1612.05644 (2)
        ["CMB_Serpico_P.txt",    "CMB PSCCK",        "k"     , 3 ],    #	arxiv: 1707.04206
        ["cmb_FLC.txt", 		 "CMB FLC22", 'darkblue',  2],		# arxiv: 2212.07969
        ["CMB_conservative.txt", "CMB AEGSSM", 'darkblue',  2],		# arxiv:  2403.18895
        ["SegueI.txt",           "Segue I" ,  "green",  2 ],
        ["EridanusII.txt",       "Eridanus II",   'm'   , 2 ],
        ["XRayB.txt",            "XRay binaries",     "orange" , 3 ],
        ["LalphaForest.txt",     r"Ly$\alpha$",  "c"  , 2 ],
        ["DynamicalFriction.txt","Dyn. friction",  'darkgreen' , 1 ],
        ["FirstClouds.txt",      "First Clouds",  'yellow'   , 2 ],
        ["UFD.txt",              "UFD" ,        'g',     3 ],
        ["WideBinaries.txt",     "Wide Binaries",   'm' , 3 ],
        ["cmb_dist.txt",         "CMB dist.",  "blue"     , 1 ],
        ["DG.txt",               "Gal. Disk",         'brown' , 1 ],
        #
        #
        #

#        ["MACHO.txt",           "MACHO",     'purple' , 2 ],
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
        # ["CMB.txt",                "CMB",       "purple"  , 3 ],
        ## ["text.txt",  , None ],   ## ??
        # ["LEoT1.txt",            "LE oT1",       'r',    2 ],
        
#
]
