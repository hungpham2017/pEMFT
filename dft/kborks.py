'''
Block-Diagonalized Embedded Mean-Field Theory for Restricted Kohn-Sham with k-point sampling
'''

import os, sys
import numpy as np
from scipy import optimize
from functools import reduce
from pyscf.pbc import gto
from pyscf.pbc.scf import khf

	
class KBORKS(khf.KRHF):
	def __init__(self, atoms, basis, pseudo, xc, subsystemA, kpts=np.zeros((1,3)), verbose=4):
		'''

		Args:
			
		Return:
		
		'''	

		self.cell = self.make_cell(atoms, basis, pseudo, verbose)
		
		self.nao = cell.nao_nr()
		self.kpts = kpts
		self.xc = xc
		
		baslst = cell.search_ao_label(ao_labels)
		cluster_A = np.zeros(cell.nao_nr())
		cluster_A[baslst] = 1
		cluster_B = np.ones(cell.nao_nr())
		cluster_B[baslst] = 0		
		cluster_A = np.matrix(cluster_A)
		cluster_B = np.matrix(cluster_B)
		self.AA = np.dot(cluster_A.T , cluster_A) == 1
		self.BB = np.dot(cluster_B.T , cluster_B) == 1
		self.AB = np.dot(cluster_A.T , cluster_B) == 1		
		self.nao_A = len(baslst)
		self.nao_B = self.nao - len(baslst)
		self.U = self.make_U()
		
	get_veff = get_veff
	
	def make_U(self):
		s_kpts = self.cell.pbc_intor('int1e_ovlp_sph', hermi=1, kpts=self.kpts)
		sAA_kpts = [s[self.AA].reshape(self.nao_A,self.nao_A) for s in s_kpts]
		sAB_kpts = [s[self.AB].reshape(self.nao_A,self.nao_B) for s in s_kpts]
		PAB_kpts = [np.linalg.inv(sAA_kpts[kpt]).dot(sAB_kpts[kpt]) for kpt in range(self.kpts.shape[0])]
		U = [np.zeros([self.nao, self.nao]) for kpt in range(self.kpts.shape[0])]
		for kpt in range(self.kpts.shape[0]):
			tlock = np.hstack((np.identity(self.nao_A), -PAB_kpts[kpt]))
			bblock = np.hstack((np.zeros([self.nao_B, self.nao_A]), np.identity(self.nao_B)))
			U[kpt] = np.vstack((tlock,bblock))
		return U
		
	def make_cell(self, atoms, basis, pseudo, verbos=4):
		'''
		
		'''
		
		cell = gto.Cell()
		cell.atom = atoms
		cell.basis = 'gth-szv'
		cell.pseudo = 'gth-pade'
		cell.verbose = 7
		cell.output = '/dev/null'
		cell.build()	
	
	
	
		