'''
Block-Diagonalized Embedded Mean-Field Theory for Restricted Kohn-Sham at a single k-point
'''
#TODO: check hf.get_grad

import time
import numpy as np
from scipy import optimize
from functools import reduce
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import gto
from pyscf.pbc.dft import rks as pbcks

def get_ovlp_AO(cell, kpt=np.zeros(3)):
	'''
	Get the AO overlap matrix.
	source: pyscf.pbc.scf.hf
	'''
	s = cell.pbc_intor('int1e_ovlp_sph', hermi=1, kpts=kpt)
	cond = np.max(lib.cond(s))
	if cond * cell.precision > 1e2:
		prec = 1e2 / cond
		rmin = max([cell.bas_rcut(ib, prec) for ib in range(cell.nbas)])
		if cell.rcut < rmin:
			logger.warn(cell, 'Singularity detected in overlap matrix.  '
						'Integral accuracy may be not enough.\n      '
						'You can adjust  cell.precision  or  cell.rcut  to '
						'improve accuracy.  Recommended values are\n      '
						'cell.precision = %.2g  or smaller.\n      '
						'cell.rcut = %.4g  or larger.', prec, rmin)
	return s

def get_ovlp(boks, cell=None, kpt=None):
	'''
	get_ovlp in BO basis
	'''
	if cell is None: cell = boks.cell
	if kpt is None: kpt = boks.kpt
	
	ovlp_AO = boks.get_ovlp_AO(cell, kpt)	#TODO: this step can be done once in the init
	return boks.AO2BO(ovlp_AO)	
		
def get_hcore(boks, cell=None, kpt=None):
	'''
	get_hcore in BO basis
	'''
	if cell is None: cell = boks.cell
	if kpt is None: kpt = boks.kpt

	from pyscf.pbc.scf import hf
	hcore_in_AO = hf.get_hcore(cell, kpt)	#TODO: this step can be done once in the init
	hcore_in_BO = boks.AO2BO(hcore_in_AO)
	return hcore_in_BO
		
def get_veff(boks, cell=None, dmBO=None, dm_last=0, vhf_last=0, hermi=1,
             kpt=None, kpts_band=None):
	'''Coulomb + XC functional

	.. note::
		This function will change the boks object.

	Args:
		boks : an instance of :class:`RKS`
			XC functional are controlled by boks.xc attribute.  Attribute
			boks.grids might be initialized.
		dmBO : ndarray or list of ndarrays
			A density matrix or a list of density matrices

	Returns:
		matrix Veff = J + Vxc.  Veff can be a list matrices, if the input
		dmBO is a list of density matrices.
	'''
	if cell is None: cell = boks.cell
	if dmBO is None: dmBO = boks.make_rdm1()
	if kpt is None: kpt = boks.kpt
	t0 = (time.clock(), time.time())
	dmAO = boks.dmBO2dmAO(dmBO) 
	dmAA = dmBO[:boks.nao_A,:boks.nao_A]
	def tofull(int_mat):
		'''Convert AA quantity to full size matrix with zeros in other blocks'''
		out_mat = np.zeros([boks.nao,boks.nao])
		out_mat[:boks.nao_A,:boks.nao_A] = int_mat
		return out_mat
		
	
	ground_state = (isinstance(dmBO, np.ndarray) and dmBO.ndim == 2
					and kpts_band is None)

# For UniformGrids, grids.coords does not indicate whehter grids are initialized
	if boks.grids.non0tab is None:
		boks.grids.build(with_non0tab=True)
		if boks.small_rho_cutoff > 1e-20 and ground_state:
			boks.grids = pbcks.prune_small_rho_grids_(boks, cell, dmAO, boks.grids, kpt)
		t0 = logger.timer(boks, 'setting up grids', *t0)

	if hermi == 2:  # because rho = 0
		n, exc, vxc = 0, 0, 0
	else:
		# E_xc^(2)[D] computed in AO, then transform to BO
		n1, exc1, vxc1_in_AO = boks._numint.nr_rks(cell, boks.grids, boks.xc[1], dmAO, 0,
										kpt, kpts_band)
		# E_xc^(1)[D^AA] computed in AO, then transform to BO	 				
		n2, exc2, vxc2_in_AO = boks._numint.nr_rks(boks.cellAA, boks.grids, boks.xc[0], dmAA, 0,
										kpt, kpts_band)
		# E_xc^(2)[D^AA] computed in AO, then transform to BO								
		n3, exc3, vxc3_in_AO = boks._numint.nr_rks(boks.cellAA, boks.grids, boks.xc[1], dmAA, 0,
										kpt, kpts_band)
		n = n1 + n2 - n3
		exc = exc1 + exc2 - exc3
		vxc_in_AO = vxc1_in_AO + tofull(vxc2_in_AO - vxc3_in_AO)
		
		logger.debug(boks, 'nelec by numeric integration = %s', n)
		t0 = logger.timer(boks, 'vxc', *t0)

	omega, alpha, hyb = boks._numint.rsh_and_hybrid_coeff(boks.xc[0], spin=cell.spin)
	if abs(hyb) < 1e-10:
		vj = boks.get_j(cell, dmAO, hermi, kpt, kpts_band)
		vxc_in_AO += vj
	else:
		if getattr(boks.with_df, '_j_only', False):  # for GDF and MDF
			boks.with_df._j_only = False
		vj = boks.get_j(cell, dmAO, hermi, kpt, kpts_band)
		if boks.EXscheme == 'EX0':
			vk_small = boks.kmfAA.get_jk(boks.cellAA, dmAA, hermi, kpt, kpts_band)[1]
			vk = tofull(vk_small)
		else:
			pass #TODO: updating new scheme
			
		vxc_in_AO += vj - vk * (hyb * .5)

		if ground_state:
			exc -= np.einsum('ij,ji', dmAA, vk_small).real * .5 * hyb * .5

	if ground_state:
		ecoul = np.einsum('ij,ji', dmAO, vj).real * .5
	else:
		ecoul = None
		
	vxc_in_BO = boks.AO2BO(vxc_in_AO)	#Transform to BO
	vxc_in_BO = lib.tag_array(vxc_in_BO, ecoul=ecoul, exc=exc, vj=None, vk=None)
	return vxc_in_BO
	
	
def energy_elec(boks, dmBO=None, h1e=None, vhf=None):
	'''
	Electronic part of RKS energy

	Args:
		boks : an instance of BOKS class

		dmBO : 2D ndarray
			one-partical density matrix in BO basis
		h1e : 2D ndarray
			Core hamiltonian

	Returns:
		RKS electronic energy and the 2-electron part contribution
	'''
	if dmBO is None: dmBO = boks.make_rdm1()
	if h1e is None: h1e = boks.get_hcore()
	if vhf is None or getattr(vhf, 'ecoul', None) is None:
		vhf = boks.get_veff(boks.cell, dmBO)
	e1 = np.einsum('ij,ji', h1e, dmBO).real
	tot_e = e1 + vhf.ecoul + vhf.exc
	logger.debug(boks, 'Ecoul = %s  Exc = %s', vhf.ecoul, vhf.exc)
	return tot_e, vhf.ecoul+vhf.exc
	

		
class BORKS(pbcks.RKS):
	def __init__(self, lattice, atoms, basis, pseudo, xc, sub_systemA, kpt=np.zeros((1,3)), verbose=4):
		'''
		Args:
			
		Return:
		
		'''	
		self.cell, self.cellAA = self.make_cell(lattice, atoms, basis, pseudo, sub_systemA, verbose)
		self.kmfAA = pbcks.RKS(self.cellAA)
		self.verbose = verbose
		pbcks.RKS.__init__(self, self.cell, kpt)		
		self.nao = self.cell.nao_nr()
		self.kpt = kpt
		self.xc = xc
		self.EXscheme = 'EX0'
		
		# Mask for EMFT blocks
		baslst = self.cell.search_ao_label(sub_systemA)
		cluster_A = np.zeros(self.nao)
		cluster_A[baslst] = 1
		cluster_B = np.ones(self.nao)
		cluster_B[baslst] = 0		
		cluster_A = np.matrix(cluster_A)
		cluster_B = np.matrix(cluster_B)
		self.AA = np.dot(cluster_A.T , cluster_A) == 1
		self.BB = np.dot(cluster_B.T , cluster_B) == 1
		self.AB = np.dot(cluster_A.T , cluster_B) == 1		
		self.nao_A = len(baslst)
		self.nao_B = self.nao - len(baslst)
		assert self.nao_A > 0
		
		# BO projector
		self.U = self.make_U(self.kpt)
		self.Uinv = np.linalg.inv(self.U)

	get_ovlp = get_ovlp
	get_hcore = get_hcore
	get_veff = get_veff
	energy_elec = energy_elec
	
	def make_U(self, kpt):
		'''
		Construct the U matrix used to transform the AO basis to BO basis
		ref: Equation (18) in JCTC2017, 10.1021/acs.jctc.6b01065
		'''
		s = self.get_ovlp_AO(self.cell, kpt=kpt)
		sAA = s[self.AA].reshape(self.nao_A,self.nao_A)
		sAB = s[self.AB].reshape(self.nao_A,self.nao_B)
		PAB = np.linalg.inv(sAA).dot(sAB)
		U = np.identity(self.nao)
		U[:self.nao_A,self.nao_A:] = -PAB
		return U
		
	def AO2BO(self, matrix):
		return reduce(np.dot, (self.U.T, matrix, self.U))
		
	def BO2AO(self, matrix):
		return reduce(np.dot, (self.Uinv.T, matrix, self.self.Uinv))	
		
	def dmBO2dmAO(self, dm_BO=None ):
		if dm_BO is None: dm_BO = self.make_rdm1()
		return reduce(np.dot, (self.U, dm_BO, self.U.T))	
		
	def get_ovlp_AO(self, cell=None, kpt=None):
		if cell is None: cell = self.cell
		if kpt is None: kpt = self.kpt
		return get_ovlp_AO(cell, kpt)
		
	def make_cell(self, lattice, atoms, basis, pseudo, sub_systemA, verbose=4):
		'''
		Create a cell instance for EMFT
		'''
		
		#Create the cell instance for the total system with dual basis
		cell = gto.Cell()
		cell.a = lattice
		cell.atom = atoms
		cell.basis = basis[1]
		cell.pseudo = pseudo[1]
		cell.verbose = 0
		cell.build()
		cell_basis = {}
		cell_pseudo = {}
		for atom in cell._atom:
			label = atom[0]
			if label in sub_systemA: 
				cell_basis[label] = basis[0]
				cell_pseudo[label] = pseudo[0]
			else:
				cell_basis[label] = basis[1]
				cell_pseudo[label] = pseudo[1]				
				
		cell.basis = cell_basis
		cell.pseudo = cell_pseudo
		cell.verbose = verbose
		cell.build()	
		
		#Create the cell instance for the subsystem A
		cellAA = gto.Cell()
		cellAA.a = lattice		
		cellAA.atom = [atom for atom in cell.atom if atom[0] in sub_systemA]
		cellAA.basis = basis[0]
		cellAA.pseudo = pseudo[0]	
		cellAA.verbose = 0		
		cellAA.build()
		
		return cell, cellAA
	
	def get_init_guess_need_to_modify(self):
		#TODO: Need to transform the guess RDM to BO basis
		pass
		
	def dmAA(self, dmBO=None):
		'''
		Extract the AA block from the 1RDM in BO basis 
		and transform it to AO basis
		'''
		if dmBO is None: dmBO = self.make_rdm1()
		
		dmAA_inBO = np.zeros_like(dmBO)
		dmAA_inBO[:self.nao_A,:self.nao_A] = dmBO[:self.nao_A,:self.nao_A] 
		dmAA_inAO = self.dmBO2dmAO(dmAA_inBO)
		return dmAA_inAO, dmBO[:self.nao_A,:self.nao_A]
			
		
		
