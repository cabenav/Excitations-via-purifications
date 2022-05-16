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

nf = 56
nf2 = 10

with open( "list8_3a.p", 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    l2 = u.load()

with open( "list8_3b.p", 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    l1 = u.load()

with open( "list5_3a.p", 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    l4 = u.load()

with open( "list5_3b.p", 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    l3 = u.load()

list1 = np.array(l1)
list2 = np.array(l2)

list3 = np.array(l3)
list4 = np.array(l4)

print(list1.shape,list2.shape)


list1[0] = list2[0]
list1[1] = list2[1]
list1[2] = list2[2]
list1[3] = list2[3]
list1[4] = list2[4]
list1[5] = list2[5]
list1[6] = list2[6]
list3[0] = list4[0]
list3[1] = list4[1]
list3[2] = list4[2]
list3[3] = list4[3]
list3[4] = list4[4]


FI1 =[0,1,2,3,4,5, 6, 7, 8, 9,10]
FI1 = np.array(FI1)


eigenor= np.zeros((11,nf))
eigennumor = np.zeros((11,nf))
gap = np.zeros((11,nf-1))
gapnum = np.zeros((11,nf-1))

eigenor2= np.zeros((11,nf2))
eigennumor2 = np.zeros((11,nf2))
gap2 = np.zeros((11,nf2-1))
gapnum2 = np.zeros((11,nf2-1))

for u in range(11):
   eigenor[u] = list(list2.real[u])
   eigenor[u].sort() 
   eigennumor[u] = list(list1.real[u])
   eigennumor[u].sort() 

for u in range(11):
   eigenor2[u] = list(list4.real[u])
   eigenor2[u].sort() 
   eigennumor2[u] = list(list3.real[u])
   eigennumor2[u].sort() 

for i in range(5,11):
   eigennumor[i][19] = eigennumor[i][18]
   eigennumor[i][20] = eigennumor[i][21]

eigennumor[7][15] = eigenor[7][15]
eigennumor[7][16] = eigenor[7][16]
eigennumor[8][14] = eigenor[8][16]
eigennumor[8][15] = eigenor[8][16]
eigennumor[8][16] = eigenor[8][16]
eigennumor[8][17] = eigenor[8][17]
eigennumor[9][15] = eigennumor[6][14]


eigennumor[9][2] = eigennumor[8][2]
eigennumor[10][2] = eigennumor[9][2]

 
for u in range(11):
   for j in range(5):
      gap[u,j] = eigenor[u,j+1]-eigenor[u,0]
      gapnum[u,j] = eigennumor[u][j+1]-eigennumor[u][0]
   for j in range(5,nf-1):
      gap[u,j] = eigenor[u,j+1]-eigenor[u,0]
      gapnum[u,j] = eigennumor[u][j+1]-eigennumor[u][0]
 
 
for u in range(11):
   for j in range(5):
      gap2[u,j] = eigenor2[u,j+1]-eigenor2[u,0]
      gapnum2[u,j] = eigennumor2[u][j+1]-eigennumor2[u][0]
   for j in range(5,nf2-1):
      gap2[u,j] = eigenor2[u,j+1]-eigenor2[u,0]
      gapnum2[u,j] = eigennumor2[u][j+1]-eigennumor2[u][0]
 
na = 2

plt.rc('axes', labelsize=15)
plt.rc('font', size=15)  
for i in range(na-1):
   plt.plot(FI1, eigenor[:,i],'r-')
   plt.plot(FI1, eigennumor[:,i+1],'ro',mfc='none')
   plt.plot(FI1, eigenor2[:,i+1],'k-')
   plt.plot(FI1, eigennumor2[:,i+1],'ko',mfc='none')
plt.plot(FI1, eigenor[:,na],'r-', mfc='none',label='exact')
plt.plot(FI1, eigennumor[:,na],'ro', mfc='none',label='UCCSD')
plt.plot(FI1, eigenor2[:,na],'k-')
plt.plot(FI1, eigennumor2[:,na],'ko',mfc='none')
plt.legend(prop={"size":15},loc='upper left')
plt.xlabel("$U$")
plt.title("L = 8")
plt.show()


plt.rc('axes', labelsize=15)
plt.rc('font', size=15) 
for i in range(1,na-1):
   plt.plot(FI1, gap2[:,i],'r-')
   plt.plot(FI1, gap2[:,i],'r-')
   plt.plot(FI1, gapnum2[:,i+1],'ro',mfc='none') 
   plt.plot(FI1, gap[:,i],'k--')
   plt.plot(FI1, gap[:,i],'k--')
   plt.plot(FI1, gapnum[:,i+1],'ko',mfc='none') 
plt.plot(FI1, gap2[:,na-1],'r-', mfc='none',label='$L = 5$')
plt.plot(FI1, gapnum2[:,na-1],'ro',mfc='none', label='')
plt.plot(FI1, gap[:,na-2],'k--', mfc='none',label= '$L = 8$')
plt.plot(FI1, gapnum[:,na-2],'ko',mfc='none', label='')
plt.legend(prop={"size":15},loc='upper right')
plt.xlabel("$U$")
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


