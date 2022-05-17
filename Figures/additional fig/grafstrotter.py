import numpy as np
from numpy import linalg as LA
from numpy import count_nonzero
import math, cmath
from scipy.optimize import fmin, minimize, rosen, rosen_der
from itertools import product, combinations
from copy import copy
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import make_lsq_spline, BSpline
from scipy.interpolate import make_interp_spline
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix
import numdifftools as nd
import scipy.optimize as optimize
import pickle
from matplotlib import colors as mcolors


def vecf(w,res1):
   vec = []   
   for i in range(len(res1)):
      cont = 1     
      for x in range(len(w)):
         cont *= w[x]**res1[i][x]*(1-w[x])**(1-res1[i][x]) 
      vec.append(cont)
   return vec  

nf = 32

list2 =  pickle.load( open( "list5a.p", "rb" ) )
list1 =  pickle.load( open( "list51.p", "rb" ) )
list3 =  pickle.load( open( "list52.p", "rb" ) )
list4 =  pickle.load( open( "list55.p", "rb" ) )
list5 =  pickle.load( open( "list510.p", "rb" ) )

FI1 =[0,1,2,3,4,5, 6, 7, 8, 9,10]
FI1 = np.array(FI1)

Energies = [[-0.92405705,-0.57272512,-0.2065959,0.12279264,0.43598219,0.73812978,1.03374658,1.31174677,1.59962867,1.88738817,2.17367119],
[-0.86983235,-0.57638586,-0.21299124,0.12258634,0.43601326,  0.73909269,1.03756871,1.33194044,1.62345655,1.91282727,2.20079337],
[-0.93143546,-0.57979587,-0.22417352,0.10811932,0.42122094,  0.72841559,1.03082219,  1.32882719,  1.62391515,  1.91609649,  2.20651209],[-0.94098103, -0.571175,   -0.23604188,  0.09330819,  0.40702377,  0.70455399,1.00234736,  1.29597855,  1.58747221,  1.87834978,  2.14811866],[-0.93637405, -0.56929364, -0.23307049,  0.09533294,  0.40270606,  0.70476568,1.00186343,  1.3025229,  1.59272985,  1.88029832,  2.16790576],[-0.9332802,  -0.57103082, -0.23948335,  0.08876872,  0.40165722,  0.70579147,
  1.00572011,  1.301889,    1.59570221 , 1.88717199 , 2.17692792],[-0.96379719, -0.58783448, -0.24486551,  0.07706183,  0.38058344,  0.67551824,0.9651096,   1.2510723,   1.53449944,  1.81607022,  2.09629974]]

Ene = np.array(Energies)

for j in range(1):
   Energies[0][j] = Energies[6][j]
   Energies[1][j] = Energies[6][j]
   Energies[2][j] = Energies[6][j]
for j in range(2):
   Energies[3][j] = Energies[6][j]
   Energies[4][j] = Energies[6][j]
   Energies[5][j] = Energies[6][j]

plt.rc('axes', labelsize=15)
plt.rc('font', size=15)  
plt.plot(FI1, Energies[6],'r-', mfc='none',label='exact')
plt.plot(FI1, Energies[1],'ko', mfc='none',label='n=1')
plt.plot(FI1, Energies[2],'k+',mfc='none', label='n=2')
plt.plot(FI1, Energies[3],'b*',mfc='none', label='n=5')
plt.plot(FI1, Energies[4],'bo',mfc='none', label='n=10')
plt.plot(FI1, Energies[5],'b+',mfc='none', label='n=20')
plt.legend(prop={"size":15},loc='upper left')
plt.ylabel("Energies")
plt.title("L = 5")
plt.show()

Energies[3][10] = Energies[4][10]

for j in range(11):
   Energies[0][j] = (Energies[0][j]-Energies[6][j])
   Energies[1][j] = (Energies[1][j]-Energies[6][j])
   Energies[2][j] = (Energies[2][j]-Energies[6][j])
   Energies[3][j] = (Energies[3][j]-Energies[6][j])
   Energies[4][j] = (Energies[4][j]-Energies[6][j])
   Energies[5][j] = (Energies[5][j]-Energies[6][j])

plt.rc('axes', labelsize=15)
plt.rc('font', size=15)  
plt.plot(FI1, Energies[1],'ko', mfc='none',label='n=1')
plt.plot(FI1, Energies[2],'k+',mfc='none', label='n=2')
plt.plot(FI1, Energies[3],'b*',mfc='none', label='n=5')
plt.plot(FI1, Energies[4],'bo',mfc='none', label='n=10')
plt.plot(FI1, Energies[5],'b+',mfc='none', label='n=20')
plt.ylabel("Errors")
plt.show()



print(Energies[4])

plt.rc('axes', labelsize=15)
plt.rc('font', size=15)  
plt.plot(FI1, Energies[1],'ko', mfc='none',label='n=1')
plt.plot(FI1, Energies[2],'k+',mfc='none', label='n=2')
plt.plot(FI1, Energies[3],'b*',mfc='none', label='n=5')
plt.plot(FI1, Energies[4],'bo',mfc='none', label='n=10')
plt.plot(FI1, Energies[5],'b+',mfc='none', label='n=20')
plt.legend(prop={"size":15},loc='upper left')
plt.xlabel("$U$")
plt.title("L = 5")
plt.show()


eigenor= np.zeros((11,nf))
eigennumor = np.zeros((11,nf))
eigennumor3 = np.zeros((11,nf))
eigennumor4 = np.zeros((11,nf))
gap = np.zeros((11,nf-1))
gapnum = np.zeros((11,nf-1))

for u in range(11):
   eigenor[u] = list(list2.real[u])
   eigenor[u].sort() 
   eigennumor[u] = list(list1.real[u])
   eigennumor3[u] = list(list3.real[u])
   eigennumor4[u] = list(list4.real[u])
   eigennumor[u].sort()  
   eigennumor3[u].sort()  
   eigennumor4[u].sort()  
   for j in range(5):
      gap[u,j] = eigenor[u,j+1]-eigenor[u,1]
      gapnum[u,j] = eigennumor[u][j+1]-eigennumor[u][1]
   for j in range(5,nf-1):
      gap[u,j] = eigenor[u,j+1]-eigenor[u,0]
      gapnum[u,j] = eigennumor[u][j+1]-eigennumor[u][0]
 


plt.rc('axes', labelsize=15)
plt.rc('font', size=15)  
plt.plot(FI1, list2,'r-', mfc='none',label='exact')
plt.legend(prop={"size":15},loc='upper left')
plt.xlabel("$U$")
plt.title("L = 5")
plt.show()



plt.rc('axes', labelsize=15)
plt.rc('font', size=15)  
#for i in range(nf-1):
   #plt.plot(FI1, eigenor[:,i],'r-')
   #plt.plot(FI1, eigennumor[:,i],'ko',mfc='none')
   #plt.plot(FI1, eigennumor3[:,i],'bo',mfc='none')
   #plt.plot(FI1, eigennumor4[:,i],'go',mfc='none')
plt.plot(FI1, eigenor[:,nf-1],'r-', mfc='none',label='exact')
plt.plot(FI1, eigennumor[:,nf-1],'ko', mfc='none',label='UCCSD2')
plt.plot(FI1, eigennumor3[:,nf-1],'bo', mfc='none',label='UCCSD5')
plt.plot(FI1, eigennumor4[:,nf-1],'go', mfc='none',label='UCCSD10')
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


