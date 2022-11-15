import numpy as np
from classes import ClassPkSpectrum 




ks = 10**np.linspace(1.0, 8.0, 100, True)

# Set Gaussian Pk
m="gaussian"
CPk = ClassPkSpectrum(pkmodel=m)
Pks = CPk.Pk(ks)

print("Gaussian Pks : " , Pks)


# Import ks and Pks
m="user_import"
CPk = ClassPkSpectrum(pkmodel=m, user_k=ks, user_Pk=Pks)
Pks_UI = CPk.Pk(ks)

print("User imported Pks : " , Pks_UI)
print("Are the same:", np.all(Pks == Pks_UI)  )



# Set own funtion
def MyPkFunction(kk): return np.ones_like(kk)*0.1
CPk.set_Pk_function(MyPkFunction)
Pks_UI = CPk.Pk(ks)

print("User imported Pks : " , Pks_UI)
print("Are the same:", np.all(Pks == Pks_UI)  )


