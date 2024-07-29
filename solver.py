import os, sys, numpy, scipy
from typing import Iterable
from pyscf import fci

class Hamiltonian(object):
    verbose = 0
    def __init__(self, norb, nelec, h1e, h2e):
        self._norb = norb
        self._h1e = h1e
        self._h2e = h2e

        if isinstance(nelec, int):
            nelec_beta = nelec // 2
            nelec_alph = nelec - nelec_beta
            self._nelec = (nelec_alph, nelec_beta)

        elif isinstance(nelec, Iterable):
            assert len(nelec) == 2
            self._nelec = (nelec[0], nelec[1])

        else:
            raise ValueError
        
        self.check_sanity()

    def check_sanity(self):
        norb = self._norb
        h1e = self._h1e
        h2e = self._h2e

        nelec_alph, nelec_beta = self._nelec
        assert self._nelec == (nelec_alph, nelec_beta)
        assert h1e.shape   == (norb, norb)
        assert h2e.shape   == (norb, norb, norb, norb)

    @property
    def norb(self):
        self.check_sanity()
        return self._norb
    
    @property
    def nsite(self):
        self.check_sanity()
        return self._norb

    @property
    def nelec(self):
        self.check_sanity()
        nelec_alph, nelec_beta = self._nelec
        return (nelec_alph, nelec_beta)
    
    @property
    def nelec_alph(self):
        self.check_sanity()
        return self._nelec[0]
    
    @property
    def nelec_beta(self):
        self.check_sanity()
        return self._nelec[1]
    
    @property
    def h1e(self):
        self.check_sanity()
        return self._h1e
    
    @property
    def h2e(self):
        self.check_sanity()
        return self._h2e

class HubbardModel(Hamiltonian):
    pass

class HubbardModel1D(HubbardModel):
    def __init__(self, norb, nelec, hop_t=1.0, hub_u=1.0, is_pbc=True):
        self._hop_t = hop_t
        self._hub_u = hub_u
        self._is_pbc = is_pbc

        from numpy import zeros, arange
        h1e = numpy.zeros((norb, norb))
        h1e[arange(norb-1),   arange(norb-1)+1] = -hop_t
        h1e[arange(norb-1)+1, arange(norb-1)]   = -hop_t
        if is_pbc:
            h1e[0, norb-1] = -hop_t
            h1e[norb-1, 0] = -hop_t

        h2e = numpy.zeros((norb, norb, norb, norb))
        for i in range(norb):
            h2e[i, i, i, i] = hub_u

        super().__init__(norb, nelec, h1e, h2e)


Hubbard1D = HubbardModel1D


def solve_fci_ene(h: Hamiltonian):
    fci_obj = fci.direct_spin1.FCI()
    fci_obj.verbose = h.verbose
    fci_obj.nroots  = 4
    fci_obj.conv_tol = 1e-12
    e, v = fci_obj.kernel(h.h1e, h.h2e, h.norb, h.nelec)
    return e.min()

def solve_fci_rdm1(h: Hamiltonian):
    fci_obj = fci.direct_spin1.FCI()
    fci_obj.verbose = h.verbose
    fci_obj.nroots  = 4
    fci_obj.conv_tol = 1e-12
    e, v = fci_obj.kernel(h.h1e, h.h2e, h.norb, h.nelec)
    e0 = e.min()
    v0 = v[e.argmin()]

    if h.verbose > 0:
        print(f"FCI Ground State Energy = {e0: 12.8f}")

    rdm1 = fci_obj.make_rdm1(v0, h.norb, h.nelec)
    return rdm1

def solve_rhf_ene(h: Hamiltonian, dm0=None):
    from pyscf import gto, scf
    m = gto.M()
    m.nelectron = h.nelec[0] + h.nelec[1]
    m.spin = h.nelec[0] - h.nelec[1]
    mf = scf.RHF(m)
    mf.verbose = 0
    mf.get_hcore = lambda *args: h.h1e
    mf.get_ovlp = lambda *args: numpy.eye(h.norb)
    mf._eri = h.h2e

    mf.kernel(dm0=dm0)
    return mf.e_tot

def solve_rhf_rdm1(h: Hamiltonian, dm0=None):
    from pyscf import gto, scf
    m = gto.M()
    m.nelectron = h.nelec[0] + h.nelec[1]
    m.spin = h.nelec[0] - h.nelec[1]
    mf = scf.RHF(m)
    mf.conv_tol = 1e-4
    mf.conv_tol_grad = 1e-4
    mf.max_cycle = 200
    mf.verbose = 4
    mf.diis_space = 2
    mf.get_hcore = lambda *args: h.h1e
    mf.get_ovlp = lambda *args: numpy.eye(h.norb)
    mf._eri = h.h2e

    if dm0 is None:
        dm0 = numpy.eye(h.norb) * m.nelectron / h.norb

    mf.kernel(dm0=dm0)
    return mf.make_rdm1()