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
import numdifftools as nd
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
L = int(input("L number (integer) of sites: "))
trotter = int(input("Trotter (integer) steps: "))
answer = input("Do you want to introduce the w-values (y/n): ")
w = list(np.arange(0.5/L,0.5+0.01,0.5/L)) 
Num = int(input("Number of particles: "))
ni = int(math.factorial(L)/(math.factorial(Num-1)*math.factorial(L-Num+1))+1)
nf = int(math.factorial(L)/(math.factorial(Num)*math.factorial(L-Num)))

if answer == "y":
   for i in range(L):
      w[i] = float(input("Please enter a weight between 0 and 0.5: "))
elif answer == "n": 
   w = list(np.arange(0.5/L,0.5+0.01,0.5/L)) 
else: 
    print("Please next time enter (y/n). I take my weights.")

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
Op = np.zeros((L+1,2**L,2**L))

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



eigen = [] 

for u in range(11):
   v1, v2 = LA.eig(Ham(Ham1,Ham2,u))
   eigen.append(v1)

eigen = np.array(eigen)

#CONSTRUCTION OF UNITARIES

#Generation of all Doubles Excitations
test_list = np.arange(0, L, 1).tolist()
res2 = list(combinations(test_list,2))
Doubles = []
for j1 in range(len(res2)):
   for k1 in range(j1+1,len(res2)):
      if(common_member(res2[j1],res2[k1])==False):
         #print(res2[j1],res2[k1],common_member(res2[j1],res2[k1]),j1,k1)
         Doubles.append((res2[j1],res2[k1]))

AllD = np.zeros((len(Doubles),nf,nf))
AllS = np.zeros((len(res),nf,nf))

for j1 in range(len(Doubles)):
   AllD[j1] = UnD(Doubles[j1][0][0],Doubles[j1][0][1],Doubles[j1][1][0],Doubles[j1][1][1])[ni:ni+nf,ni:ni+nf]

for j1 in range(len(res2)):
   AllS[j1] = UnS(res2[j1][0],res2[j1][1])[ni:ni+nf,ni:ni+nf]


FI1 =[0,1,2,3,4,5, 6, 7, 8, 9,10]
FI1 = np.array(FI1)


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

def Unit(params,res2,Hamil):
   x = params
   Full = np.identity(nf)
   Full1 = np.identity(nf)
   FullS = np.identity(nf)
   Full1S = np.identity(nf)
   for j1 in range(len(Doubles)):
      Full = np.matmul(Unitary(x[j1],AllD[j1]),Full)
      Full1 = np.matmul(Full1, Unitary(-x[j1],AllD[j1]))
   for j1 in range(len(Doubles),2*len(Doubles)):
      j11 =  j1-len(Doubles)
      Full = np.matmul(Unitary(x[j1],AllD[j11]),Full)
      Full1 = np.matmul(Full1, Unitary(-x[j1],AllD[j11]))
   for j1 in range(2*len(Doubles),2*len(Doubles)+len(res2)):
      j11 = j1-2*len(Doubles)
      FullS = np.matmul(Unitary(x[j1],AllS[j11]),FullS)
      Full1S = np.matmul(Full1S, Unitary(-x[j1],AllS[j11]))
   for j1 in range(2*len(Doubles)+len(res2),2*len(Doubles)+2*len(res2)):
      j11 = j1-2*len(Doubles)-len(res2)
      FullS = np.matmul(Unitary(x[j1],AllS[j11]),FullS)
      Full1S = np.matmul(Full1S, Unitary(-x[j1],AllS[j11]))
   Full = np.matmul(LA.matrix_power(FullS,trotter),np.matmul(LA.matrix_power(Full, trotter),FullS))
   Full1 = np.matmul(np.matmul(Full1S,LA.matrix_power(Full1, trotter)),LA.matrix_power(Full1S,trotter))
   #Full = np.matmul(LA.matrix_power(np.matmul(FullS,Full),trotter),FullS)
   #Full1 = np.matmul(Full1S,LA.matrix_power(np.matmul(Full1,Full1S),trotter))
   return np.matmul(np.matmul(Full1,Hamil),Full)


def gradient_descent(gradient, start, learn_rate, n_iter, tolerance):
   vector = start
   for _ in range(n_iter):
      diff = -learn_rate * gradient(vector).real
      if np.all(np.abs(diff) <= tolerance):
         break
      vector += diff
   return vector


weights = vecf(w,res1)
eigennum = np.zeros((11,nf))
eigenor= np.zeros((11,nf))
eigennumor = np.zeros((11,nf))
gap = np.zeros(11)
gapnum = np.zeros(11)
gap2 = np.zeros(11)
gapnum2 = np.zeros(11)

seed=list(np.full(2*len(Doubles)+2*len(res2),0))
Hamil=Ham(Ham1,Ham2,0)

for u in range(11):
   print("I am computing the energies for the coupling u: ", u)
   Hamil=Ham(Ham1,Ham2,u)
   fun = function(weights[ni:ni+nf],res2,Hamil)
   #seed = gradient_descent(fun.grad,seed,0.2,5,5e-03)
   seed = optimize.fmin(fun.evalua, seed,maxfun=100000,maxiter=100000,ftol=5e-5,xtol=5e-5)
   vec=np.zeros(nf)
   vecaux=np.zeros(nf)
   for i in range(nf):
      vec=np.zeros(nf)
      vec[i]=1
      eigennum[u,i] = np.matmul(np.matmul(vec,Unit(seed,res2,Hamil)),vec)
   print(eigennum[u])
   print(eigen[u])
   eigenor[u] = list(eigen.real[u])
   eigenor[u].sort() 
   eigennumor[u] = list(eigennum.real[u])
   eigennumor[u].sort()   
   gap[u] = eigenor[u,1]-eigenor[u,0]
   gapnum[u] = eigennumor[u][1]-eigennumor[u][0]
   gap2[u] = eigenor[u,2]-eigenor[u,0]
   gapnum2[u] = eigennumor[u][2]-eigennumor[u][0]
 

pickle.dump(eigen, open( "list53a.p", "wb" ) )
pickle.dump(eigennum, open( "list53b.p", "wb" ) )


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

plt.rc('axes', labelsize=15)
plt.rc('font', size=15)  
plt.plot(FI1, gap,'bo', mfc='none',label='exact')
plt.plot(FI1, gapnum,'r*',label='UCC')
plt.legend(prop={"size":15},loc='upper left')
plt.xlabel("$U/t$")
plt.show()

plt.rc('axes', labelsize=15)
plt.rc('font', size=15)  
plt.plot(FI1, gap2,'bo', mfc='none',label='exact')
plt.plot(FI1, gapnum2,'r*',label='UCC')
plt.legend(prop={"size":15},loc='upper left')
plt.xlabel("$U/t$")
plt.show()

