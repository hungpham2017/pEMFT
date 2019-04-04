'''
Block-Diagonalized Embedded Mean-Field Theory for Restricted Kohn-Sham at a single k-point
'''

import time
import numpy as np

def to_cartesian(lattice_mat, atoms_fractional):
	atoms = atoms_fractional.split() 
	lattice = np.sqrt((np.sum(lattice_mat**2, axis = 1)))
	atoms_cartesian = []
	for atom in atoms:
		symbol, x, y, z = atom.split('0.')
		x = np.float64('0.'+ x)
		y = np.float64('0.'+ y)
		z = np.float64('0.'+ z)
		x, y, z = np.asarray([x,y,z]) * lattice
		atoms_cartesian.append([symbol, [x,y,z]])
	return atoms_cartesian

