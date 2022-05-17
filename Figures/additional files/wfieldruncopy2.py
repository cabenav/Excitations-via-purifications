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
 
#FUNCTIONS

def dimensionH(Nn):
   cont = 0
   for i in range (Nn):
      cont += math.factorial(L)/(math.factorial(i)*math.factorial(L-i))
   return int(cont)


def expf(x,L):
   return complex(np.cos(2*np.pi*x/L),np.sin(2*np.pi*x/L))

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True 
    else:
        return False

def Ham(H1,H2,U):
   return (H1+np.multiply(H2,U))[ni:ni+nf,ni:ni+nf]

def UnD(a1,a2,a3,a4):
   return np.matmul(np.matmul(np.matmul(np.transpose(Op[a1]),np.transpose(Op[a2])),Op[a3]),Op[a4])-np.matmul(np.matmul(np.matmul(np.transpose(Op[a4]),np.transpose(Op[a3])),Op[a2]),Op[a1])

def UnS(a1,a2):
   return np.matmul(np.transpose(Op[a1]),Op[a2])-np.matmul(np.transpose(Op[a2]),Op[a1])

def Unitary(th,OpAux):
   return np.identity(nf)+np.multiply(OpAux,np.sin(th))-np.multiply((np.cos(th)-1),np.matmul(OpAux,OpAux))


def vecf(w,res1):
   vec = []   
   for i in range(len(res1)):
      cont = 1     
      for x in range(len(w)):
         cont *= w[x]**res1[i][x]*(1-w[x])**(1-res1[i][x]) 
      vec.append(cont)
   return vec       
      
class function():
   def __init__(self, weig,res,Hamil):
      self.res = res
      self.Hamil = Hamil
      self.weig = weig
   def evalua(self,seed):
      matrizSD = Unit(seed,self.res,self.Hamil)
      elem = 0
      for ji in range(len(self.weig)):
         vec=np.zeros(len(self.weig))
         vec[ji]=1
         elem += self.weig[ji]*np.matmul(np.matmul(vec,matrizSD),vec)
      return elem
   def grad(self,seed):
      return nd.Gradient(self.evalua)(seed)
    

#NUMBER OF SITES, WEIGHTS and TROTTER STEPS:
L = 5
trotter = 2
w = list(np.arange(0.5/L,0.5+0.01,0.5/L)) 
Num = 2
ni = 0
nf = 32

#GENERATION OF THE HILBERT (FOCK) SPACE
## This generates the Hilbert space {|000>,|001>,...} but in a non-organized way
res = [ele for ele in product([0,1], repeat = L)]
res1 = [ele for ele in product([0,0], repeat = L)]
res = np.array(res)
res1 = np.array(res1)

#ORGANIZING THIS IN N-PARTICLE SECTORS
##res1 contains ALL the states in the Hilbert space
sumsec= np.sum(res, axis=1)
vec = []
for j in range(L+1):
   vec.append([i for i,x in enumerate(sumsec) if x == j])

order= sum(vec,[])
for j in range(2**L):
   res1[j] = res[order[j]]


#GENERATION OF THE ANHILITATION OPERATORS
##Op[0],Op[1]... are the anhilitation operators for sites 0,1,...
Op = np.zeros((L+1,2**L,2**L))
OpS = []

for j in range(L):
   res2 = [ele for ele in product([0,0], repeat = L)]
   res2 = copy(res1)
   res2[:,j] = np.zeros(2**L)
   aux2 = copy(res1[:,j])
   if j == 0:
      sta= copy(res1[:,0])
   else:
      sta = [math.pow(-1,value) for value in sum(res1[:,x] for x in range(j))] 
   for j1 in range(2**L):
      for j2 in range(2**L):
         if np.array_equal(res1[j1],res2[j2]):
            Op[j,j1,j2] = sta[j2]*aux2[j2]

Op[L] = Op[0]

##This is optional to verify the fermionic commutation rules:
#print(np.sum(Op[0],axis=1))
#print(np.diag(np.matmul(np.transpose(Op[0]), Op[0])+np.matmul(Op[0],np.transpose(Op[0]))))


#CONSTRUCTION OF THE HAMILTONIANS

Ham1 =-sum(np.multiply(expf((k1-k2)*jj,L)*expf(-k2,L)/L,np.matmul(np.transpose(Op[k1]), Op[k2]))+np.multiply(expf((k1-k2)*jj,L)*expf(k1,L)/L,np.matmul(np.transpose(Op[k1]), Op[k2])) for k1 in range(L) for k2 in range(L) for jj in range(L))

Ham2 =sum(np.multiply(expf((k1-k2+k3-k4)*jj,L)*expf(k3-k4,L)/L**2,np.matmul(np.matmul(np.matmul(np.transpose(Op[k1]), Op[k2]),np.transpose(Op[k3])),Op[k4])) for k1 in range(L) for k2 in range(L) for k3 in range(L) for k4 in range(L) for jj in range(L))

#Ham1S =-sum(np.multiply(expf((k1-k2)*jj,L)*expf(-k2,L)/L,np.multiply(csr_matrix.transpose(OpS[k1]), OpS[k2]))+np.multiply(expf((k1-k2)*jj,L)*expf(k1,L)/L,np.multiply(csr_matrix.transpose(OpS[k1]), OpS[k2])) for k1 in range(L) for k2 in range(L) for jj in range(L))

#Ham2S =sum(np.multiply(expf((k1-k2+k3-k4)*jj,L)*expf(k3-k4,L)/L**2,np.multiply(np.multiply(np.multiply(csr_matrix.transpose(OpS[k1]), OpS[k2]),csr_matrix.transpose(OpS[k3])),OpS[k4])) for k1 in range(L) for k2 in range(L) for k3 in range(L) for k4 in range(L) for jj in range(L))


eigen = []
eigen1 = []
eigenor1 = np.zeros((11,5))
wor1 = np.zeros(5)
eigen2 = []
eigenor2 = np.zeros((11,10))
wor2 = np.zeros(10)
eigen3 = []
eigenor3 = np.zeros((11,10)) 
wor3 = np.zeros(10)
eigen4 = []
eigenor4 = np.zeros((11,5)) 
wor4 = np.zeros(5)
eigen5 = []
eigenor5 = np.zeros((11,1)) 
wor5 = np.zeros(1)
entan = []



for u in range(11):
   v1, v2 = LA.eig(Ham(Ham1,Ham2,u)[1:6,1:6])
   eigen1.append(v1)
   v1, v2 = LA.eig(Ham(Ham1,Ham2,u)[6:16,6:16])
   eigen2.append(v1)
   v1, v2 = LA.eig(Ham(Ham1,Ham2,u)[16:26,16:26])
   eigen3.append(v1)
   v1, v2 = LA.eig(Ham(Ham1,Ham2,u)[26:31,26:31])
   eigen4.append(v1)
   v1, v2 = LA.eig(Ham(Ham1,Ham2,u)[31:32,31:32])
   eigen5.append(v1)
   v1, v2 = LA.eig(Ham(Ham1,Ham2,u))
   eigen.append(v1)
   eigen[u] = eigen[u].real 
   prov = []
   for s in range(len(v2[:,1])):
      prov.append(sum([abs(num)**4 for num in v2[:,s]]))
   entan.append(prov)
   ordering= sorted(range(len(eigen[u])), key=lambda k: eigen[u].real[k])
   eigen[u].real.sort()
   aa = copy(entan[u])
   for s in range(len(entan[u])):
      aa[s] = entan[u][ordering[s]]
   entan[u] = aa
  

eigen = np.array(eigen)
eigen1 = np.array(eigen1)
eigen2 = np.array(eigen2)
eigen3 = np.array(eigen3)
eigen4 = np.array(eigen4)
eigen5 = np.array(eigen5)

w[2] = 0.2
exact = np.zeros(20)
u = 5

print(w)
for ss in range(20):
   w[2] += 0.01
   print(w)
   weights = vecf(w,res1)
   weights = np.array(weights)
   wor1 = list(weights[1:6])
   wor1.sort(reverse = True)
   wor2 = list(weights[6:16])
   wor2.sort(reverse = True)
   wor3 = list(weights[16:26])
   wor3.sort(reverse = True)
   wor4 = list(weights[26:31])
   wor4.sort(reverse = True)
   wor5 = list(weights[31:32])
   wor5.sort(reverse = True)
   eigenor1[u] = list(eigen1.real[u])
   eigenor1[u].sort()
   exact[ss] +=np.dot(eigenor1[u],wor1) 
   eigenor2[u] = list(eigen2.real[u])
   eigenor2[u].sort()
   exact[ss] +=np.dot(eigenor2[u],wor2)
   eigenor3[u] = list(eigen3.real[u])
   eigenor3[u].sort() 
   exact[ss] +=np.dot(eigenor3[u],wor3)
   eigenor4[u] = list(eigen4.real[u])
   eigenor4[u].sort()
   exact[ss] +=np.dot(eigenor4[u],wor4)
   eigenor5[u] = list(eigen5.real[u])
   eigenor5[u].sort() 
   exact[ss] +=np.dot(eigenor5[u],wor5)  



FI1 =w = list(np.arange(0.2,0.4,0.01))
FI1 = np.array(FI1)
print(FI1)


plt.rc('axes', labelsize=15)
plt.rc('font', size=15)  
plt.plot(FI1, exact,'r-', color='blue',mfc='none',label='exact $\mathcal{E}(w)$',markersize=8)
plt.legend(prop={"size":15},loc='upper left')
plt.show()

