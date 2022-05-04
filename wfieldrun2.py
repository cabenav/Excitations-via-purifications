import numpy as np
from numpy import linalg as LA
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
import scipy.optimize as optimize
import pickle
 
#FUNCTIONS

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
   return H1+np.multiply(H2,U)

def UnD(th,a1,a2,a3,a4,Od):
   OpAux = np.matmul(np.matmul(np.matmul(np.transpose(Od[a1]),np.transpose(Od[a2])),Od[a3]),Od[a4])-np.matmul(np.matmul(np.matmul(np.transpose(Od[a4]),np.transpose(Od[a3])),Od[a2]),Od[a1])
   OpAux = OpAux[ni:ni+nf,ni:ni+nf]
   return np.identity(nf)+np.multiply(OpAux,np.sin(th))-np.multiply((np.cos(th)-1),np.matmul(OpAux,OpAux))

def UnS(th,a1,a2,Od):
   OpAux = np.matmul(np.transpose(Od[a1]),Od[a2])-np.matmul(np.transpose(Od[a2]),Od[a1])
   OpAux = OpAux[ni:ni+nf,ni:ni+nf]
   return np.identity(nf)+np.multiply(OpAux,np.sin(th))-np.multiply((np.cos(th)-1),np.matmul(OpAux,OpAux))


def vecf(w,res1):
   vec = []   
   for i in range(len(res1)):
      cont = 1     
      for x in range(len(w)):
         cont *= w[x]**res1[i][x]*(1-w[x])**(1-res1[i][x]) 
      vec.append(cont)
   return vec       
      
         

#NUMBER OF SITES, WEIGHTS and TROTTER STEPS:
L = 6
trotter = 20
w = list(np.arange(0.5/L,0.5+0.01,0.5/L)) 
Num = 2
ni = int(math.factorial(L)/(math.factorial(Num-1)*math.factorial(L-Num+1))+1)
nf = int(math.factorial(L)/(math.factorial(Num)*math.factorial(L-Num)))

round_to_w = [round(num, 3) for num in w]
print("************************")
print("************************")
print("The weights are: ", round_to_w)
print("Now the calculations start (relax)")
print("****************************")
print("*   *   *  *** * *** *   *** ")
print(" * * * *   **  * **  *   * * ")
print("  *   *    *   * *** *** *** ")
print("***************************")


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
Op = np.zeros((L,2**L,2**L))

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

##This is optional to verify the fermionic commutation rules:
#print(np.sum(Op[0],axis=1))
#print(np.diag(np.matmul(np.transpose(Op[0]), Op[0])+np.matmul(Op[0],np.transpose(Op[0]))))


#CONSTRUCTION OF THE HAMILTONIANS

Ham1 =-sum(np.multiply(expf((k1-k2)*jj,L)*expf(-k2,L)/L,np.matmul(np.transpose(Op[k1]), Op[k2]))+np.multiply(expf((k1-k2)*jj,L)*expf(k1,L)/L,np.matmul(np.transpose(Op[k1]), Op[k2])) for k1 in range(L) for k2 in range(L) for jj in range(L))

Ham2 =sum(np.multiply(expf((k1-k2+k3-k4)*jj,L)*expf(k3-k4,L)/L**2,np.matmul(np.matmul(np.matmul(np.transpose(Op[k1]), Op[k2]),np.transpose(Op[k3])),Op[k4])) for k1 in range(L) for k2 in range(L) for k3 in range(L) for k4 in range(L) for jj in range(L))

#EIGENVALUES AND w IN DECREASING ORDER

eigen = []
wnew = copy(w)
v1, v2 = LA.eig(Ham(Ham1,Ham2,0)[1:L+1,1:L+1])
eigen.append(v1)
eigen = np.array(eigen)
#print(eigen.real)
order = np.argsort(eigen.real)
#print(w)

for z in range(L): 
   wnew[order[0][L-1-z]] = w[z]

w = copy(wnew)

eigen = [] 

for u in range(11):
   v1, v2 = LA.eig(Ham(Ham1,Ham2,u)[ni:ni+nf,ni:ni+nf])
   eigen.append(v1)

eigen = np.array(eigen)

#print(np.sum(UnD(np.pi/2,1,1,2,2,Op),axis=1))
#[L+1:int(L+L*(L-1)/2),L+1:L+int(L*(L-1)/2)])

FI1 =[0,1,2,3,4,5, 6, 7, 8, 9,10]
FI1 = np.array(FI1)
#print(eigen) 

plt.rc('axes', labelsize=15)
plt.rc('font', size=15)  
for i in range(nf):
   plt.plot(FI1, eigen[:,i], 'bo', mfc='none')
plt.xlabel("$U/t$")


#QUANTUM ALGORITHM: here starts the quantum calculation

#Generation of all Doubles Excitations
test_list = np.arange(0, L, 1).tolist()
res2 = list(combinations(test_list,2))
Doubles = []
for j1 in range(len(res2)):
   for k1 in range(j1+1,len(res2)):
      if(common_member(res2[j1],res2[k1])==False):
         #print(res2[j1],res2[k1],common_member(res2[j1],res2[k1]),j1,k1)
         Doubles.append((res2[j1],res2[k1]))

def Unit(params,Doubles,res2,Hamil,Od,trotter):
   x = params
   Full = np.identity(nf)
   Full1 = np.identity(nf)
   FullS = np.identity(nf)
   Full1S = np.identity(nf)
   for j1 in range(len(Doubles)):
      Full = np.matmul(UnD(x[j1],Doubles[j1][0][0],Doubles[j1][0][1],Doubles[j1][1][0],Doubles[j1][1][1],Od),Full)
      Full1 = np.matmul(Full1, UnD(-x[j1],Doubles[j1][0][0],Doubles[j1][0][1],Doubles[j1][1][0],Doubles[j1][1][1],Od))
   for j1 in range(len(Doubles),2*len(Doubles)):
      j11 =  j1-len(Doubles)
      Full = np.matmul(UnD(x[j1],Doubles[j11][0][0],Doubles[j11][0][1],Doubles[j11][1][0],Doubles[j11][1][1],Od),Full)
      Full1 = np.matmul(Full1, UnD(-x[j1],Doubles[j11][0][0],Doubles[j11][0][1],Doubles[j11][1][0],Doubles[j11][1][1],Od))
   for j1 in range(2*len(Doubles),2*len(Doubles)+len(res2)):
      j11 = j1-2*len(Doubles)
      FullS = np.matmul(UnS(x[j1],res2[j11][0],res2[j11][1],Od),FullS)
      Full1S = np.matmul(Full1S, UnS(-x[j1],res2[j11][0],res2[j11][1],Od))
   for j1 in range(2*len(Doubles)+len(res2),2*len(Doubles)+2*len(res2)):
      j11 = j1-2*len(Doubles)-len(res2)
      FullS = np.matmul(UnS(x[j1],res2[j11][0],res2[j11][1],Od),FullS)
      Full1S = np.matmul(Full1S, UnS(-x[j1],res2[j11][0],res2[j11][1],Od))
   Full = np.matmul(LA.matrix_power(FullS,trotter),np.matmul(LA.matrix_power(Full, trotter),FullS))
   Full1 = np.matmul(np.matmul(Full1S,LA.matrix_power(Full1, trotter)),LA.matrix_power(Full1S,trotter))
   #Full = np.matmul(LA.matrix_power(np.matmul(FullS,Full),trotter),FullS)
   #Full1 = np.matmul(Full1S,LA.matrix_power(np.matmul(Full1,Full1S),trotter))
   return np.matmul(np.matmul(Full1,Hamil[ni:ni+nf,ni:ni+nf]),Full)

def function(seed,weights,Doubles,res2,Ham,Op,trotter):
   matrizSD = Unit(seed,Doubles,res2,Ham,Op,trotter)
   elem = 0
   #for ji in range(len(weights)):
   for ji in range(nf):
      vec=np.zeros(nf)
      vec[ji]=1
      elem += weights[ji+ni]*np.matmul(np.matmul(vec,matrizSD),vec)
   return elem
   
   

weights = vecf(w,res1)
seed=list(np.full(2*len(Doubles)+2*len(res2),0))

eigennum = np.zeros((11,nf))
for u in range(11):
   print("I am computing the energies for the coupling u: ", u)
   seed = optimize.fmin(function, seed,args=(weights,Doubles,res2,Ham(Ham1,Ham2,u),Op,trotter),maxfun=1000000,maxiter=1000000,ftol=1e-4,xtol=1e-4)
   vec=np.zeros(nf)
   vecaux=np.zeros(nf)
   for i in range(nf):
      vec=np.zeros(nf)
      vec[i]=1
      eigennum[u,i] = np.matmul(np.matmul(vec,Unit(seed,Doubles,res2,Ham(Ham1,Ham2,u),Op,trotter)),vec)
   #print(eigennum)

pickle.dump(eigen, open( "list10.p", "wb" ) )
pickle.dump(eigennum, open( "list20.p", "wb" ) )

plt.rc('axes', labelsize=15)
plt.rc('font', size=15)  
for i in range(nf-1):
   plt.plot(FI1, eigen[:,i],'bo', mfc='none')
   plt.plot(FI1, eigennum[:,i],'r*')
plt.plot(FI1, eigen[:,nf-1],'bo', mfc='none',label='exact')
plt.plot(FI1, eigennum[:,nf-1],'r*', label='UCC')
plt.legend(prop={"size":15},loc='upper left')
plt.xlabel("$U/t$")
plt.show()
