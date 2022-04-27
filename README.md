# Excitations-via-purifications
This is the code of the paper "Excitations of quantum many-body systems via purified ensembles" https://arxiv.org/abs/2201.10974

The main code is wfield.py. This code diagonalizes the spinless Fermi-Hubbard model with L number of sites both analitically and with the quantum computating algorithm of the paper. The diagonalization is performed in the entire Fock space, i.e. N = 0,1,2,...,L-1,L, and the results are presented in each N particle sector. 

Since L is the number of sites this is the main variable of the code.
The second variable is the number of Trotter steps.
