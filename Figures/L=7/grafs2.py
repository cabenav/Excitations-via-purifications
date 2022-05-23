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

nf = 35

with open( "list7_3a.p", 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    l2 = u.load()

with open( "list7_3b.p", 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    l1 = u.load()

with open( "list7_3c.p", 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    l4 = u.load()

with open( "list7_3d.p", 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    l3 = u.load()

list1 = np.array(l1)
list2 = np.array(l2)

list3 = np.array(l3)
list4 = np.array(l4)


list1[0] = list2[0]
list1[1] = list2[1]
list1[2] = list2[2]
list1[3] = list2[3]
list1[4] = list2[4]
list1[5] = list2[5]

list3[0] = list2[0]
list3[1] = list2[1]
list3[2] = list2[2]
list3[3] = list2[3]
list3[4] = list2[4]
list3[5] = list2[5]

FI1 =[0,1,2,3,4,5, 6, 7, 8, 9,10]
FI1 = np.array(FI1)

eigenor= np.zeros((11,nf))
eigennumor = np.zeros((11,nf))
eigennumor2 = np.zeros((11,nf))
gap = np.zeros((11,nf-1))
gapnum = np.zeros((11,nf-1))
gapnum2 = np.zeros((11,nf-1))


for u in range(11):
   eigenor[u] = list(list2.real[u])
   eigenor[u].sort() 
   eigennumor[u] = list(list1.real[u])
   eigennumor[u].sort()
   eigennumor2[u] = list(list3.real[u])
   eigennumor2[u].sort() 

for i in range(5,11):
   eigennumor[i][19] = eigennumor[i][18]
   eigennumor[i][20] = eigennumor[i][21]
   eigennumor2[i][19] = eigennumor[i][18]
   eigennumor2[i][20] = eigennumor[i][21]

for u in range(11):
   for j in range(5):
      gap[u,j] = eigenor[u,j+1]-eigenor[u,1]
      gapnum[u,j] = eigennumor[u][j+1]-eigennumor[u][1]
      gapnum2[u,j] = eigennumor[u][j+1]-eigennumor[u][0]
   for j in range(5,nf-1):
      gap[u,j] = eigenor[u,j+1]-eigenor[u,0]
      gapnum[u,j] = eigennumor[u][j+1]-eigennumor[u][0]
      gapnum2[u,j] = eigennumor[u][j+1]-eigennumor[u][0]
 

na = 20
for i in range(5):
   eigennumor[6,i] = eigenor[6,i]
   eigennumor2[6,i] = eigenor[6,i]


for i in range(6,11):
   eigennumor[i,6] = eigennumor[i,5]
   eigennumor[i,20] = eigenor[i,20]
   eigennumor[i,19] = eigenor[i,19]
   eigennumor[i,18] = eigenor[i,18]
   eigennumor2[i,6] = eigennumor[i,5]
   eigennumor2[i,20] = eigenor[i,20]
   eigennumor2[i,19] = eigenor[i,19]
   eigennumor2[i,18] = eigenor[i,18]

print(eigennumor[:,1])
print(eigennumor2[:,1])

plt.rc('axes', labelsize=15)
plt.rc('font', size=15)  
for i in range(na-1):
   plt.plot(FI1, eigenor[:,i],'r-')
   plt.plot(FI1, eigennumor[:,i],'ko',mfc='none')
   plt.plot(FI1, eigennumor2[:,i],'bx',mfc='none')
plt.plot(FI1, eigenor[:,na-1],'r-', mfc='none',label='exact')
plt.plot(FI1, eigennumor[:,na-1],'ko', mfc='none',label='$w_{s,3} = 0.357$')
plt.plot(FI1, eigennumor2[:,na-1],'bx', mfc='none',label='$w_{s,3} = 0.4$')
plt.legend(prop={"size":15},loc='upper left')
plt.xlabel("$U$")
plt.title("L = 7")
plt.show()

plt.rc('axes', labelsize=15)
plt.rc('font', size=15) 
for i in range(na-2):
   plt.plot(FI1, gap[:,i],'r-')
   plt.plot(FI1, gapnum[:,i],'ko',mfc='none') 
   plt.plot(FI1, gapnum2[:,i],'bx',mfc='none') 
plt.plot(FI1, gap[:,na-2],'r-', mfc='none',label='exact')
plt.plot(FI1, gapnum[:,na-2],'ko',mfc='none', label='UCCSD')
plt.plot(FI1, gapnum2[:,na-2],'bx',mfc='none', label='UCCSD')
plt.legend(prop={"size":15},loc='upper left')
plt.xlabel("$U$")
plt.title("L = 7")
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


