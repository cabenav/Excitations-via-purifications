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

with open( "list3.p", 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    list2 = u.load()

with open( "list4.p", 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    list1 = u.load()
 
with open( "list1.p", 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    list4 = u.load()

with open( "list2.p", 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    list3 = u.load()

FI1 =[0,1,2,3,4,5, 6, 7, 8, 9,10]
FI1 = np.array(FI1)

list1[0] = list2[0]
list1[1] = list2[1]
list1[2] = list2[2]
list1[3] = list2[3]
list3[0] = list4[0]
list3[1] = list4[1]
list3[2] = list4[2]
list3[3] = list4[3]

eigenor= np.zeros((11,nf))
eigennumor = np.zeros((11,nf))
eigenor1= np.zeros((11,nf))
eigennumor1 = np.zeros((11,nf))
gap = np.zeros((11,nf-1))
gapnum = np.zeros((11,nf-1))

for u in range(11):
   eigenor[u] = list(list2.real[u])
   eigenor[u].sort() 
   eigennumor[u] = list(list1.real[u])
   eigennumor[u].sort() 
   eigenor1[u] = list(list4.real[u])
   eigenor1[u].sort() 
   eigennumor1[u] = list(list3.real[u])
   eigennumor1[u].sort()  
   for j in range(5):
      gap[u,j] = eigenor[u,j+1]-eigenor[u,1]
      gapnum[u,j] = eigennumor[u][j+1]-eigennumor[u][1]
   for j in range(5,nf-1):
      gap[u,j] = eigenor[u,j+1]-eigenor[u,0]
      gapnum[u,j] = eigennumor[u][j+1]-eigennumor[u][0]
 
plt.rc('axes', labelsize=15)
plt.rc('font', size=15)  
for i in range(nf-1):
   plt.plot(FI1, eigenor[:,i],'r-')
   plt.plot(FI1, eigennumor[:,i],'ko',mfc='none')
   plt.plot(FI1, eigennumor1[:,i],'bo',mfc='none')
plt.plot(FI1, eigenor[:,nf-1],'r-', mfc='none',label='exact')
plt.plot(FI1, eigennumor[:,nf-1],'ko', mfc='none',label='$w_{s,2}=0.28$')
plt.plot(FI1, eigennumor1[:,nf-1],'bo', mfc='none',label='$w_{s,2}=0.4$')
plt.legend(prop={"size":15},loc='upper left')
plt.xlabel("$U$")
plt.title("L = 5")
plt.show()

plt.rc('axes', labelsize=15)
plt.rc('font', size=15) 
for i in range(nf-2):
   plt.plot(FI1, gap[:,i],'r-')
   plt.plot(FI1, gapnum[:,i],'ko',mfc='none') 
plt.plot(FI1, gap[:,nf-2],'r-', mfc='none',label='exact')
plt.plot(FI1, gapnum[:,nf-2],'ko',mfc='none', label='UCCSD')
plt.legend(prop={"size":15},loc='upper left')
plt.xlabel("$U$")
plt.title("L = 5")
plt.show()

def func(vec, vec1):
   disT=0
   for j in range(len(vec)):
      dis = 1
      for i in range(len(vec1)):
        # if (float(abs((vec[j].real - vec1[i].real)/vec1[i].real)))**2 < dis:
         #    dis = (float(abs((vec[j].real - vec1[i].real)/vec1[i].real)))**2
         if (float(abs((vec[j].real - vec1[i].real)))) < dis:
             dis = (float(abs((vec[j].real - vec1[i].real))))
      disT += dis
   return disT/len(vec1)

error = np.zeros(11)
errorT = 0
for i in range(11):
   error[i] = func(gap[i],gapnum[i])   
   print(func(gap[i],gapnum[i]))
   errorT += error[i]

print("Total error: ", errorT/11)

plt.plot(FI1, error,'ko',mfc='none',label='error')
plt.legend(prop={"size":15},loc='upper left')
plt.xlabel("$U/t$")
plt.show()


