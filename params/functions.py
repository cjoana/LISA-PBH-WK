from functools import wraps

##################################
# Class decorators 
##################################

# Decorator to prevent adding new attributes after initiation
def prevent_new_attrs(cls):
    cls.__frozen = False

    def frozensetattr(self, key, value):
        if self.__frozen and not hasattr(self, key):
            mess = "Class {0} doesn't have attribute `{1}`.\n"
            mess += "Cannot set {1} = {2}"
            mess = mess.format(cls.__name__, key, value)
            print(mess)
            raise AttributeError(mess)
        else:
            object.__setattr__(self, key, value)

    def init_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self.__frozen = True
        return wrapper

    cls.__setattr__ = frozensetattr
    cls.__init__ = init_decorator(cls.__init__)

    return cls


#################################
#  Other functions
##################################

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