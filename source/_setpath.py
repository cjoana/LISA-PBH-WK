import sys, os
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
SOURCEPATH = os.path.abspath(os.path.join(ROOTPATH, 'source'))
PARAMSPATH = os.path.abspath(os.path.join(ROOTPATH, 'params'))
sys.path.append(ROOTPATH)
sys.path.append(SOURCEPATH)
sys.path.append(PARAMSPATH)

if __name__ == "__main__":

    mess = "You are setting and adding the following paths: \n"
    mess += f"  ROOTPATH = {ROOTPATH} \n"
    mess += f"  SOURCEPATH = {SOURCEPATH} \n"
    mess += f"  PARAMSPATH = {PARAMSPATH} \n"
    print(mess + "Good bye.\n")

