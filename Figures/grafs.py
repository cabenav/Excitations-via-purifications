import numpy as np
import random
import pickle
import gzip as gz
import csv
from copy import copy
import matplotlib.pyplot as plt
import pickle
from scipy import interpolate
from scipy.interpolate import make_lsq_spline, BSpline
from scipy.interpolate import make_interp_spline
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from scipy.interpolate import interp1d

nf = 10

list1a =  pickle.load( open( "Figure_5_25_1.p", "rb" ) )
list1b =  pickle.load( open( "Figure_5_25_2.p", "rb" ) )
list2a =  pickle.load( open( "Exact_5_2.p", "rb" ) )

list1c = copy(list1a)
list1c[1,:] = list1b[1,:]
list1c[5,:] = list1b[5,:]
list1c[7,:] = list1b[7,:]
list1c[8,:] = list1b[8,:]
list1c[9,:] = list1b[9,:]
list1 = np.zeros((10,10))
list2 = np.zeros((10,10))

for i in range(len(list1c)-1):
   list1[i] = list1c[i]
   list2[i] = list2a[i]

print(list1)

FI1 =[0,1,2,3,4,5, 6, 7, 8, 9]
FI1 = np.array(FI1)

plt.rc('axes', labelsize=15)
plt.rc('font', size=15)  
for i in range(nf-2):
   plt.plot(FI1, list2[:,i],'bo', mfc='none')
   plt.plot(FI1, list1[:,i],'r*')
plt.plot(FI1, list2[:,nf-2],'bo', mfc='none',label='exact')
plt.plot(FI1, list1[:,nf-2],'r*', label='UCC')
plt.legend(prop={"size":15},loc='upper left')
plt.xlabel("$U/t$")
plt.show()
