import numpy as np
import random
import pickle
import gzip as gz
import csv
import math, cmath
from copy import copy
import matplotlib.pyplot as plt
import pickle
from scipy import interpolate
from scipy.interpolate import make_lsq_spline, BSpline
from scipy.interpolate import make_interp_spline
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from scipy.interpolate import interp1d

nf = int(math.factorial(6)/(math.factorial(2)*math.factorial(4)))

list2 =  pickle.load( open( "list1.p", "rb" ) )
list1 =  pickle.load( open( "list2.p", "rb" ) )

FI1 =[0,1,2,3,4,5, 6, 7, 8, 9,10]
list1[1,:] = list2[1,:]
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

def func(vec, vec1):
   disT=0
   for j in range(len(vec)):
      dis = 1
      for i in range(len(vec1)):
         if (float(abs((vec[j].real - vec1[i].real)/vec1[i].real)))**2 < dis:
            dis = (float(abs((vec[j].real - vec1[i].real)/vec1[i].real)))**2
         #if (float(abs((vec[j].real - vec1[i].real)))) < dis:
          #   dis = (float(abs((vec[j].real - vec1[i].real)/vec1[i].real)))**2
      disT += dis
   return np.sqrt(disT)/len(vec1)

error = np.zeros(10)
errorT = 0
for i in range(nf-1):
   error[i] = func(list1[i],list2[i])   
   print(func(list1[i],list2[i]))
   errorT += error[i]

print("Total error: ", errorT/10)

plt.plot(FI1, error,'bo', mfc='none',label='error')
plt.legend(prop={"size":15},loc='upper left')
plt.xlabel("$U/t$")
plt.show()
