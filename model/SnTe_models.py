import numpy as np
import functools as ft
import kwant
import tinyarray as ta
from .lcao import L_matrices, lcao_term
import scipy.linalg as la

sigma_0 = np.array([[1, 0], [0, 1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

sigmas = np.array([sigma_x, sigma_y, sigma_z])

# Parameters listed for the two sublattices
SnTe_6band_params = dict(t=2*np.array([[-0.25, 0.45], [0.45, 0.25]]), # hoppings
                         m=np.array([-1.65, 1.65]),                   # on-site energies
                         lam = np.array([-0.3, -0.3]),                # on-site SoC's
                        )

def SnTe_6band(translations=None):
    """Make bulk 6-band model of SnTe from https://arxiv.org/pdf/1202.1003.pdf

    translations: kwant.lattice.TranslationalSymmetry
        Optional different translational unit cell. Should still be a primitive UC.
    """

    L = L_matrices()
    ### TODO: there may be a 1/2 missing from the spin here
    L_dot_S = np.sum([np.kron(sigmas[i], L[i]) for i in range(3)], axis=0)

    def onsite(site, m, lam):
        # which sublattice
        a = np.sum(site.tag) % 2
        os = m[a] * np.eye(6)
        # L dot S onsite SoC
        spinorb = lam[a] * L_dot_S
        # x, y, z = site.pos
        os = os + spinorb
        return os

    def hopping(site1, site2, t):
        # which sublattice
        a = np.sum(site1.tag) % 2
        b = np.sum(site2.tag) % 2
        d = site1.tag - site2.tag
        # ppsigma bonding
        if d<=1
        d = np.array(d) / la.norm(d)
        else 
        d1 = np.array(d)/(np.sqrt(2+2*np.cos(theta)*np.sin(theta))
        d2 = np.array(d)/(np.sqrt(2-2*np.cos(theta)*np.sin(theta))
        d1 = np.outer(d1,d1)
        d2 = np.outer(d2,d2)
        dtd = np.kron(np.eye(2), d)
        # Use the appropriate hopping depending on sublattices
        hop = t[a, b] * dtd
        return hop

    # Cubic rocksalt structure with FCC symmetry
    lat = kwant.lattice.general(np.eye(3), norbs=6)
    if translations is None:
        # Default translation vectors of FCC structure
        translations = kwant.lattice.TranslationalSymmetry([1, 1, 0], [1, 0, 1], [0, 1, 1])
    syst = kwant.Builder(symmetry=translations)

    # Populate the builder using the cubic lattice sites
    # Two sublattices
    syst[lat(0, 0, 0)] = onsite
    syst[lat(0, 0, 1)] = onsite

    # First and second neighbor hoppings
    syst[lat.neighbors(1)] = hopping
    syst[lat.neighbors(2)] = hopping

    return syst


# Define 18-orbital model including spinful s, p, d
# orbitals from Lent et.al. Superlattices and
# Microstructures, Vol. 2, Issue 5, 491-499, (1986).
# Sign choice is consistent with extra minus signs in vps and vsp
# this is only a gauge transformation adding - sign
# to all s orbitals
SnTe_18band_params = dict(
            esc= -6.578,
            esa= -12.067,
            epc= 1.659,
            epa= -0.167,
            edc= 8.38,
            eda= 7.73,
            lambdac= 0.592,
            lambdaa= 0.564,
            vss= -0.510,
            vsp= -1*0.949,
            vps= -1*-0.198,
            vpp= 2.218,
            vpppi= -0.446,
            vpd= -1.11,
            vpdpi= 0.624,
            vdp= -1.67,
            vdppi= 0.766,
            vdd= -1.72,
            vdddelta= 0.618,
            vddpi=0,
            )


PbTe_18band_params = dict(
            esc= -7.612,
            esa= -11.002,
            epc= 3.195,
            epa= -0.237,
            edc= 7.73,
            eda= 7.73,
            lambdac= 1.500,
            lambdaa= 0.428,
            vss= -0.474,
            vsp= 0.705*-1,
            vps= 0.633*-1,
            vpp= 2.066,
            vpppi= -0.430,
            vpd= -1.29,
            vpdpi= 0.835,
            vdp= -1.59,
            vdppi= 0.531,
            vdd= -1.35,
            vdddelta= 0.668,
            vddpi=0,
            )


def SnTe_18band(translations=None):
    """Make bulk 18-band model of SnTe from Lent et.al.
    Superlattices and Microstructures, Vol. 2, Issue 5, 491-499, (1986).

    translations: kwant.lattice.TranslationalSymmetry
        Optional different translational unit cell. Should still be a primitive UC.
    """

    L = L_matrices()
    L_dot_S = np.sum([np.kron(0.5 * sigmas[i], L[i]) for i in range(3)], axis=0)

    @ft.lru_cache(100)
    def H_ac(d, vss, vsp, vps, vpp, vpppi, vpd, vpdpi,
             vdp, vdppi, vdd, vdddelta, vddpi):
        d = ta.array(d)
        Hac = np.zeros((18, 18), dtype=complex)
        Hac[:2, :2] = vss * np.kron(np.eye(2), lcao_term(0, 0, 0, d))
        Hac[2:8, :2] = vsp * np.kron(np.eye(2), lcao_term(1, 0, 0, d))
        Hac[:2, 2:8] = vps * np.kron(np.eye(2), lcao_term(0, 1, 0, d))
        Hac[2:8, 2:8] = (vpp * np.kron(np.eye(2), lcao_term(1, 1, 0, d))
                        + vpppi * np.kron(np.eye(2), lcao_term(1, 1, 1, d)))
        Hac[8:18, 2:8] = (vpd * np.kron(np.eye(2), lcao_term(2, 1, 0, d))
                         + vpdpi * np.kron(np.eye(2), lcao_term(2, 1, 1, d)))
        Hac[2:8, 8:18] = (vdp * np.kron(np.eye(2), lcao_term(1, 2, 0, d))
                         + vdppi * np.kron(np.eye(2), lcao_term(1, 2, 1, d)))
        Hac[8:18, 8:18] = (vdd * np.kron(np.eye(2), lcao_term(2, 2, 0, d))
                          + vddpi * np.kron(np.eye(2), lcao_term(2, 2, 1, d))
                          + vdddelta * np.kron(np.eye(2), lcao_term(2, 2, 2, d)))
        return Hac

    @ft.lru_cache(100)
    def H_os(es, ep, ed, lam):
        H = np.zeros((18, 18), dtype=complex)
        H[:2, :2] = es * np.eye(2)
        H[2:8, 2:8] = ep * np.eye(6) + lam * L_dot_S
        H[8:18, 8:18] = ed * np.eye(10)
        return H

    def onsite(site, esa, epa, eda, lambdaa, esc, epc, edc, lambdac):
        if np.sum(site.tag) % 2 == 0:
            return H_os(esa, epa, eda, lambdaa)
        else:
            return H_os(esc, epc, edc, lambdac)

    def hopping(site1, site2, vss, vsp, vps, vpp, vpppi, vpd, vpdpi,
             vdp, vdppi, vdd, vdddelta, vddpi):
        # convert to tinyarray for caching
        d = ta.array(site2.pos - site1.pos)
        if np.allclose(d, np.round(d)):
            d = ta.array(np.round(d), int)
        if np.isclose(np.sum(site1.tag) % 2, 0):
            return H_ac(d, vss, vsp, vps, vpp, vpppi, vpd, vpdpi,
                        vdp, vdppi, vdd, vdddelta, vddpi)
        else:
            return H_ac(-d, vss, vsp, vps, vpp, vpppi, vpd, vpdpi,
                        vdp, vdppi, vdd, vdddelta, vddpi).T.conj()

    # Cubic rocksalt structure with FCC symmetry
    lat = kwant.lattice.general(np.eye(3), norbs=18)
    if translations is None:
        # Default translation vectors of FCC structure
        translations = kwant.lattice.TranslationalSymmetry([1, 1, 0], [1, 0, 1], [0, 1, 1])
    syst = kwant.Builder(symmetry=translations)

    # Populate the builder using the cubic lattice sites
    # Two sublattices
    syst[lat(0, 0, 0)] = onsite
    syst[lat(0, 0, 1)] = onsite

    # First neighbor hoppings
    syst[lat.neighbors(1)] = hopping

    return syst
