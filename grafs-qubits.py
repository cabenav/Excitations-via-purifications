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



with open( "list.p", 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    lista = u.load()

lista1 = np.array(lista)
n, bins, patches = plt.hist(lista1[:,0],bins='auto')

#plt.ylabel('Probability')
plt.title('Error gs')
plt.xlim(-0.0001, 0.004)
#plt.ylim(0, 500)
#plt.grid(True)
plt.show()

