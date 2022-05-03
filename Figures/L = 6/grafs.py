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

nf = 28

list2 =  pickle.load( open( "list62.p", "rb" ) )
list1 =  pickle.load( open( "list62.p", "rb" ) )

print(list2)
print(list1)

FI1 =[0,1,2,3,4,5, 6, 7, 8, 9,10]
FI1 = np.array(FI1)



eigenor= np.zeros((11,nf))
eigennumor = np.zeros((11,nf))
gap = np.zeros(11)
gapnum = np.zeros(11)
gap2 = np.zeros(11)
gapnum2 = np.zeros(11)
gap3 = np.zeros(11)
gapnum3 = np.zeros(11)
gap4 = np.zeros(11)
gapnum4 = np.zeros(11)
gap5 = np.zeros(11)
gapnum5 = np.zeros(11)
gap6 = np.zeros(11)
gapnum6 = np.zeros(11)
gap7 = np.zeros(11)
gapnum7 = np.zeros(11)
gap8 = np.zeros(11)
gapnum8 = np.zeros(11)

for u in range(11):
   eigenor[u] = list(list2.real[u])
   eigenor[u].sort() 
   eigennumor[u] = list(list1.real[u])
   eigennumor[u].sort()   
   gap[u] = eigenor[u,1]-eigenor[u,0]
   gapnum[u] = eigennumor[u][1]-eigennumor[u][0]
   gap2[u] = eigenor[u,2]-eigenor[u,0]
   gapnum2[u] = eigennumor[u][2]-eigennumor[u][1]
   gap3[u] = eigenor[u,3]-eigenor[u,0]
   gapnum3[u] = eigennumor[u][3]-eigennumor[u][0]
   gap4[u] = eigenor[u,4]-eigenor[u,0]
   gapnum4[u] = eigennumor[u][4]-eigennumor[u][0]
   gap5[u] = eigenor[u,5]-eigenor[u,0]
   gapnum5[u] = eigennumor[u][5]-eigennumor[u][0]
   gap6[u] = eigenor[u,6]-eigenor[u,0]
   gapnum6[u] = eigennumor[u][6]-eigennumor[u][0]
   gap7[u] = eigenor[u,7]-eigenor[u,0]
   gapnum7[u] = eigennumor[u][7]-eigennumor[u][0]
   gap8[u] = eigenor[u,8]-eigenor[u,0]
   gapnum8[u] = eigennumor[u][8]-eigennumor[u][0]
 

plt.rc('axes', labelsize=15)
plt.rc('font', size=15)  
for i in range(nf-2):
   plt.plot(FI1, eigenor[:,i],'r--')
   plt.plot(FI1, eigennumor[:,i],'ko')
plt.plot(FI1, eigenor[:,nf-2],'r--', mfc='none',label='exact')
plt.plot(FI1, eigennumor[:,nf-2],'ko', label='UCCSD')
plt.legend(prop={"size":15},loc='upper left')
plt.xlabel("$U/t$")
plt.show()

plt.rc('axes', labelsize=15)
plt.rc('font', size=15)  
plt.plot(FI1, gap,'r--')
plt.plot(FI1, gapnum,'ko') 
plt.plot(FI1, gap2,'r--')
plt.plot(FI1, gapnum2,'ko')
plt.plot(FI1, gap3,'r--')
plt.plot(FI1, gapnum3,'ko')
plt.plot(FI1, gap4,'r--')
plt.plot(FI1, gapnum4,'ko') 
plt.plot(FI1, gap5,'r--')
plt.plot(FI1, gapnum5,'ko') 
plt.plot(FI1, gap6,'r--')
plt.plot(FI1, gapnum6,'ko') 
plt.plot(FI1, gap7,'r--')
plt.plot(FI1, gapnum7,'ko') 
plt.plot(FI1, gap8,'r--', mfc='none',label='exact')
plt.plot(FI1, gapnum8,'ko',label='UCCSD') 
plt.legend(prop={"size":15},loc='upper left')
plt.xlabel("$U/t$")
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
   error[i] = func(list1[i],list2[i])   
   print(func(list1[i],list2[i]))
   errorT += error[i]

print("Total error: ", errorT/11)

plt.plot(FI1, error,'kX',label='error')
plt.legend(prop={"size":15},loc='upper left')
plt.xlabel("$U/t$")
plt.show()


