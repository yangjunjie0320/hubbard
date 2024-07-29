import numpy, scipy
from solver import Hamiltonian, HubbardModel, HubbardModel1D
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

def solve_dmrg_ene(h: HubbardModel):
    assert isinstance(h, HubbardModel)
    nsite = h.nsite
    nelec_alph, nelec_beta = h.nelec
    spin = nelec_alph - nelec_beta

    hub_u = h._hub_u
    hop_t = h._hop_t

    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=4)
    driver.initialize_system(n_sites=nsite, n_elec=nelec_alph + nelec_beta, spin=spin)

    b = driver.expr_builder()
    b.add_term("cd", numpy.array([[[i, i + 1], [i + 1, i]] for i in range(nsite - 1)]).flatten(), -hop_t)
    b.add_term("CD", numpy.array([[[i, i + 1], [i + 1, i]] for i in range(nsite - 1)]).flatten(), -hop_t)
    if h._is_pbc:
        b.add_term("cd", numpy.array([[0, nsite - 1], [nsite - 1, 0]]).flatten(), -hop_t)
        b.add_term("CD", numpy.array([[0, nsite - 1], [nsite - 1, 0]]).flatten(), -hop_t)

    b.add_term("cdCD", numpy.array([[i, ] * 4 for i in range(nsite)]).flatten(), hub_u)
    b = b.finalize()

    mpo = driver.get_mpo(b, iprint=h.verbose)
    bond_dims = [20] * 4 + [40] * 4 + [80] * 4 + [160] * 4 + [320] * 4
    noises = [1e-2] * 4 + [1e-4] * 4 + [1e-6] * 4 + [0]
    thrds = [1e-10] * 8

    ket = driver.get_random_mps(tag="KET", bond_dim=bond_dims[0], nroots=1)
    e0 = driver.dmrg(
        mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises,
        thrds=thrds, cutoff=0, iprint=h.verbose
        )
    return e0

def solve_dmrg_rdm1(h: HubbardModel):
    assert isinstance(h, HubbardModel)
    nsite = h.nsite
    nelec_alph, nelec_beta = h.nelec
    spin = nelec_alph - nelec_beta

    hub_u = h._hub_u
    hop_t = h._hop_t

    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=4)
    driver.initialize_system(n_sites=nsite, n_elec=nelec_alph + nelec_beta, spin=spin)

    b = driver.expr_builder()
    b.add_term("cd", numpy.array([[[i, i + 1], [i + 1, i]] for i in range(nsite - 1)]).flatten(), -hop_t)
    b.add_term("CD", numpy.array([[[i, i + 1], [i + 1, i]] for i in range(nsite - 1)]).flatten(), -hop_t)
    if h._is_pbc:
        b.add_term("cd", numpy.array([[0, nsite - 1], [nsite - 1, 0]]).flatten(), -hop_t)
        b.add_term("CD", numpy.array([[0, nsite - 1], [nsite - 1, 0]]).flatten(), -hop_t)

    b.add_term("cdCD", numpy.array([[i, ] * 4 for i in range(nsite)]).flatten(), hub_u)
    b = b.finalize()

    mpo = driver.get_mpo(b, iprint=h.verbose)
    bond_dims = [20] * 4 + [40] * 4 + [80] * 4 + [160] * 4 + [320] * 4
    noises = [1e-2] * 4 + [1e-4] * 4 + [1e-6] * 4 + [0]
    thrds = [1e-10] * 8

    ket = driver.get_random_mps(tag="KET", bond_dim=bond_dims[0], nroots=1)
    e0 = driver.dmrg(
        mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises,
        thrds=thrds, cutoff=0, iprint=h.verbose
        )
    
    if h.verbose > 0:
        print("DMRG Energy = %12.8f" % e0)

    rdm1 = driver.get_npdm(
        ket, pdm_type=1, iprint=h.verbose,
    )

    return rdm1[0] + rdm1[1]

def solve_dmrg_gfn(h: HubbardModel, omega=None, eta=1e-2):
    assert isinstance(h, HubbardModel)
    nsite = h.nsite
    nelec_alph, nelec_beta = h.nelec
    spin = nelec_alph - nelec_beta

    hub_u = h._hub_u
    hop_t = h._hop_t

    symm_type = SymmetryTypes.SZ | SymmetryTypes.CPX
    driver = DMRGDriver(scratch="./tmp", symm_type=symm_type, n_threads=4)
    driver.initialize_system(n_sites=nsite, n_elec=nelec_alph + nelec_beta, spin=spin)

    b = driver.expr_builder()
    b.add_term("cd", numpy.array([[[i, i + 1], [i + 1, i]] for i in range(nsite - 1)]).flatten(), -hop_t)
    b.add_term("CD", numpy.array([[[i, i + 1], [i + 1, i]] for i in range(nsite - 1)]).flatten(), -hop_t)
    if h._is_pbc:
        b.add_term("cd", numpy.array([[0, nsite - 1], [nsite - 1, 0]]).flatten(), -hop_t)
        b.add_term("CD", numpy.array([[0, nsite - 1], [nsite - 1, 0]]).flatten(), -hop_t)

    b.add_term("cdCD", numpy.array([[i, ] * 4 for i in range(nsite)]).flatten(), hub_u)
    b = b.finalize()

    mpo = driver.get_mpo(b, iprint=h.verbose)
    bond_dims = [20] * 4 + [40] * 4 + [80] * 4 + [160] * 4 + [320] * 4
    noises = [1e-2] * 4 + [1e-4] * 4 + [1e-6] * 4 + [0]
    thrds = [1e-10] * 8

    ket = driver.get_random_mps(tag="KET", bond_dim=bond_dims[0], nroots=1)
    e0 = driver.dmrg(
        mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises,
        thrds=thrds, cutoff=0, iprint=h.verbose
        )
    
    hip = 1.0 * mpo
    hip.const_e -= e0

    if h.verbose > 0:
        print("DMRG Ground State Energy = %12.8f" % e0)

    gip = numpy.zeros((len(omega), nsite, nsite), dtype=numpy.complex128)
    gea = numpy.zeros((len(omega), nsite, nsite), dtype=numpy.complex128)
    des = [driver.get_site_mpo(op="d", site_index=i, iprint=0) for i in range(nsite)]
    cre = [driver.get_site_mpo(op="c", site_index=i, iprint=0) for i in range(nsite)]

    for iw, w in enumerate(omega):
        for i in range(nsite):
            des_i = driver.get_site_mpo(op="d", site_index=i, iprint=0)
            rhs = driver.get_random_mps(tag="RHS", bond_dim=400, center=ket.center, target=des[i].op.q_label + ket.info.target)
            driver.multiply(rhs, des[i], ket, n_sweeps=20, bond_dims=[400], thrds=[1e-10] * 10, iprint=h.verbose)

            zip = driver.copy_mps(rhs, tag="BRA")
            gip[iw, i, i] = driver.greens_function(
                zip, hip, des[i], ket, w, -eta, n_sweeps=20,
                bra_bond_dims=[400], ket_bond_dims=[400], 
                thrds=[1E-6] * 10, iprint=h.verbose
                )
            
            for j in range(nsite):
                if j == i:
                    continue
                cre_j = driver.get_site_mpo(op="c", site_index=j, iprint=0)
                gip[iw, i, j] = driver.expectation(
                    ket, cre_j, zip
                )

    return numpy.asarray([gip, gea]).transpose(1, 0, 2, 3)

if __name__ == "__main__":
    # for hub_u in [1.0, 4.0, 8.0]:
    #     print(f"\nHubbard U = {hub_u: 8.4f}")
    #     nsite = 8
    #     for nelec in range(nsite + 1):
    #         hub = HubbardModel1D(nsite, nelec, hub_u=hub_u, is_pbc=True)
    #         hub.verbose = 0
    #         ene_dmrg = solve_dmrg_ene(hub) / nsite / (- hub._hop_t)
    #         ene_fci = solve_fci_ene(hub) / nsite / (- hub._hop_t)
    #         err = abs(ene_dmrg - ene_fci)
    #         print(f"Filling = {nelec:2d} / {nsite}, DMRG = {ene_dmrg: 12.8f}, FCI = {ene_fci: 12.8f}, Error = {err: 6.4e}")
    #         assert err < 1e-8

    # for hub_u in [1.0, 4.0, 8.0]:
    #     print(f"\nHubbard U = {hub_u: 8.4f}")
    #     nsite = 4
    #     nelec = nsite
    #     # seems to suffer from degenerate ground state 
    #     hub = HubbardModel1D(nsite, nelec, hub_u=hub_u, is_pbc=True)
    #     hub.verbose = 0
        
    #     rdm1_dmrg = solve_dmrg_rdm1(hub)
    #     rdm1_fci = solve_fci_rdm1(hub)
    #     err = abs(rdm1_dmrg - rdm1_fci).max()

    #     import sys
    #     print(f"Filling = {nelec:2d} / {nsite}, Error = {err: 6.4e}")
    #     print("DMRG")
    #     numpy.savetxt(sys.stdout, rdm1_dmrg, fmt="% 6.4f", delimiter=", ")
    #     print("FCI")
    #     numpy.savetxt(sys.stdout, rdm1_fci, fmt="% 6.4f", delimiter=", ")
    #     assert err < 1e-5

    for hub in [1.0, 4.0, 8.0]:
        print(f"\nHubbard U = {hub: 8.4f}")
        nsite = 4
        nelec = nsite

        hub = HubbardModel1D(nsite, nelec, hub_u=hub, is_pbc=True)
        hub.verbose = 0

        from solver import solve_fci_gfn
        omega = [0.0] # numpy.linspace(-5, 5, 100)
        gfn_dmrg = solve_dmrg_gfn(hub, omega=omega, eta=1e-2)[:, 0][0]
        gfn_fci = solve_fci_gfn(hub, omega=omega, eta=1e-2)[:, 0][0]

        import sys
        print("DMRG")
        numpy.savetxt(sys.stdout, gfn_dmrg.real, fmt="% 6.4f", delimiter=", ")

        print("FCI")
        numpy.savetxt(sys.stdout, gfn_fci.real, fmt="% 6.4f", delimiter=", ")

        print("DMRG")
        numpy.savetxt(sys.stdout, gfn_dmrg.imag, fmt="% 6.4f", delimiter=", ")

        print("FCI")
        numpy.savetxt(sys.stdout, gfn_fci.imag, fmt="% 6.4f", delimiter=", ")

        err = abs(gfn_dmrg - gfn_fci).max()
        print(f"Error = {err: 6.4e}")   

