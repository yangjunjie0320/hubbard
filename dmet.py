import os, sys, numpy, scipy

from solver import Hamiltonian
from solver import solve_fci_ene, solve_fci_rdm1
from solver import solve_rhf_ene, solve_rhf_rdm1

def dmet(h : Hamiltonian, imp_ind : list = [0, 1]):
    nsite = h.nsite
    nelec = h.nelec
    h1e = h.h1e
    h2e = h.h2e

    # 1. solve the mean-field problem
    rdm1_ll = solve_rhf_rdm1(h)
    f1e = h1e + numpy.einsum("mnkl,kl->mn", h2e, rdm1_ll, optimize=True)
    f1e -= numpy.einsum("mknl,kl->mn", h2e, rdm1_ll, optimize=True) * 0.5
    
    # 2. build the embedding space
    nimp = len(imp_ind)
    env_ind = [i for i in range(nsite) if i not in imp_ind]

    nenv = nsite - nimp
    neo = 2 * len(imp_ind)
    coeff_emb = numpy.eye(nsite)[:, :neo]
    assert coeff_emb.shape == (nsite, neo)

    u, s, vh = scipy.linalg.svd(rdm1_ll[env_ind][:, imp_ind], full_matrices=False)
    coeff_emb[nimp:, nimp:] = u

    rdm1_ll_emb = numpy.einsum("mn,mp,nq->pq", rdm1_ll, coeff_emb, coeff_emb, optimize=True)
    f1e_emb = numpy.einsum("mn,mp,nq->pq", f1e, coeff_emb, coeff_emb, optimize=True)
    h2e_emb = numpy.einsum("mnkl,mp,nq,kr,ls->pqrs", h2e, coeff_emb, coeff_emb, coeff_emb, coeff_emb, optimize=True)
    h1e_emb = f1e_emb - numpy.einsum("mnkl,kl->mn", h2e_emb, rdm1_ll_emb, optimize=True)
    h1e_emb += numpy.einsum("mknl,kl->mn", h2e_emb, rdm1_ll_emb, optimize=True) * 0.5

    # 3. solve the impurity problem
    nelec_emb = numpy.einsum("ii->", rdm1_ll_emb)
    nelec_emb = numpy.round(nelec_emb).astype(int)
    hemb = Hamiltonian(neo, int(nelec_emb), h1e_emb, h2e_emb)
    rdm1_hl_emb = solve_fci_rdm1(hemb)

    rdm1_hl = numpy.zeros((nsite, nsite))
    r = rdm1_hl_emb[:nimp, nimp:] @ coeff_emb[nimp:, nimp:].T * 0.5
    for ifrag in range(nsite // nimp):
        i1 = ifrag * nimp
        i2 = (ifrag + 1) * nimp
        rdm1_hl[i1:i2, i1:i2] = rdm1_hl_emb[:nimp, :nimp]
        
        r1 = r[:, :(neo - i1)]
        r2 = r[:, (neo - i1):]
        rdm1_hl[i1:i2, i2:] += r1
        rdm1_hl[i1:i2, :i1] += r2
        rdm1_hl[i2:, i1:i2] += r1.T
        rdm1_hl[:i1, i1:i2] += r2.T

    return rdm1_hl

if __name__ == '__main__':
    from solver import Hubbard1D
    h = Hubbard1D(6, 6, hub_u=4.0)
    rdm1_dmet = dmet(h, [0, 1])
    rdm1_fci = solve_fci_rdm1(h)

    print("rdm1_dmet")
    numpy.savetxt(sys.stdout, rdm1_dmet, fmt="% 6.4f", delimiter=", ")

    print("rdm1_fci")
    numpy.savetxt(sys.stdout, rdm1_fci, fmt="% 6.4f", delimiter=", ")