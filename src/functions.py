def path_not_defined():
    raise Exception("!!! You have not set up your directory paths in user_params.py")

def getpath_datadir():
    try:
        global datadir
        return datadir
    except:
        path_not_defined()

def getpath_zetacr():
    try:
        global zetacr_file
        return zetacr_file
    except:
        path_not_defined()