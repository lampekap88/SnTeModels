# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 17:34:26 2023

@author: sdewi
"""
import kwant as kwant
import kwant.continuum

# So our matrices look nice
import sympy
sympy.init_printing()

# So we can plot things
from matplotlib import pyplot as plt

# general numerics
import numpy as np

# useful definitions for later
sigma_0 = np.array([[1, 0], [0, 1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])


# =============================================================================
# Construct Hamiltonian (BHZ)
# =============================================================================
bhz_continuum = '''
    mu * kron(sigma_0, sigma_0)
    + m * kron(sigma_0, sigma_z)
    - B * (k_x**2 + k_y**2) * kron(sigma_0, sigma_z) 
    + v * (k_x * kron(sigma_y, sigma_x) - k_y * kron(sigma_x, sigma_x))
    + vp * k_z*kron(sigma_0,sigma_y)
'''
kwant.continuum.sympify(bhz_continuum)  # Not discretized yet; just to see what it looks like

# =============================================================================
# Tight binding 
# =============================================================================
tb_matrix_elements, coords = kwant.continuum.discretize_symbolic(bhz_continuum)

bhz_model = kwant.continuum.build_discretized(tb_matrix_elements, coords)
print(bhz_model.lattice.norbs)
print(bhz_model.symmetry.periods) 


# =============================================================================
# Geometry/scattering region: 2D device
# =============================================================================
syst = kwant.Builder() # We'll fill this with the TB model of our scattering region
zlen,ylen,xlen=5,15,5
def scattering_region(site):
    x, y, z = site.tag  # position in integer lattice coordinates
    return (0 <= x <= xlen  
            and (0 <= y <= ylen )
            and (0 <= z <= zlen ))

# Fill our scattering region with the BHZ model in a certain region of space
syst.fill(template=bhz_model, shape=scattering_region, start=(0, 10,0))
# =============================================================================
# Add leads
# =============================================================================
#QSHI leads:
# lead_x = kwant.Builder(symmetry=kwant.TranslationalSymmetry((-1, 0)),
#                        time_reversal=np.kron(1j * sigma_y, sigma_0),
#                        conservation_law=np.kron(-sigma_z, sigma_0),  # spin conservation
#                       )

# lead_x.fill(bhz_model, lambda site: 0 <= site.tag[1] <= ylen, (0, 1))
# Just the same as the x-direction
# lead_y = kwant.Builder(symmetry=kwant.TranslationalSymmetry((1, 0)),
#                        time_reversal=np.kron(1j * sigma_y, sigma_0),
#                        conservation_law=np.kron(-sigma_z, sigma_0),
#                       )
# lead_y.fill(bhz_model, lambda site: 0 <= site.tag[1] <= ylen, (-1, 1));

#metallic leads

def lead_shape(site):
    (x,y,z) = site.pos
    return 0<=y<=ylen and 0<=z<=zlen

lead_left = kwant.Builder(symmetry=kwant.TranslationalSymmetry((-1, 0,0)),
                        time_reversal=np.kron(1j * sigma_y, sigma_0),
                        conservation_law=np.kron(-sigma_z, sigma_0),  # spin conservation
                      )
lead_left.fill(bhz_model.substituted(mu='mu_left_lead'), lead_shape, (0, 0,0))

lead_right = kwant.Builder(symmetry=kwant.TranslationalSymmetry((1, 0, 0)),
                        time_reversal=np.kron(1j * sigma_y, sigma_0),
                        conservation_law=np.kron(-sigma_z, sigma_0),
                      )

lead_right.fill(bhz_model.substituted(mu='mu_right_lead'), lead_shape, (-1, 1,0));

syst.attach_lead(lead_right)
syst.attach_lead(lead_left)

fsyst = syst.finalized()  # Transform the system into an efficient form for numerics
kwant.plot(fsyst)
print(fsyst)

# =============================================================================
# Compute bands / dispersion relation
# =============================================================================
topo_params = dict(m=-0.3, B=-0.18, v=0.3, vp=0.3, mu=0, mu_right_lead=0, mu_left_lead=0)  # Parameters for topological phase
trivial_params = dict(m=0.3, B=-0.18, v=0.3, vp=0.3, mu=0, mu_right_lead=0, mu_left_lead=0)  # Parameters for trivial phase

fig_bands,ax_bands=plt.subplots(1,2)
kwant.plotter.bands(fsyst.leads[0], params=topo_params,ax=ax_bands[0])
ax_bands[0].hlines(0, -np.pi, np.pi, linestyles='--')
ax_bands[0].set_xlim( -1,1)
ax_bands[0].set_ylim( -0.5,0.5)
kwant.plotter.bands(fsyst.leads[0], params=trivial_params,ax=ax_bands[1])
ax_bands[1].hlines(0, -np.pi, np.pi, linestyles='--')
ax_bands[1].set_xlim( -1,1)
ax_bands[1].set_ylim( -0.5,0.5)

# # =============================================================================
# # Modes at zero energy E=0 compute and visualize
# # =============================================================================
# propagating_modes, _ = fsyst.leads[0].modes(energy=0, params=topo_params)
# print(propagating_modes.wave_functions.shape)
# print(propagating_modes.velocities)
# print(propagating_modes.momenta)

# # Just a convenience function to plot
# def plot_densities(*densities, title='Particle'):
#     for i, rho in enumerate(densities):     
#         plt.plot(rho, 'o-', label='incoming mode ' + str(i))
#     plt.title(title + ' density for lead cross-section')
#     plt.xlabel('site')
#     if title == 'Particle':
#         plt.ylabel('$|φ_i|^2$')
#     elif title == 'Spin':
#         plt.ylabel('$φ_i^\dagger (σ_z ⊗ σ_0) φ_i$')
#     plt.legend()


# phi0, phi1 = propagating_modes.wave_functions[:, :2].transpose()

# density = kwant.operator.Density(fsyst.leads[0]) # calculate |φ_i|**2 for each site (summing degrees of freedom)
# density = density.bind(params=topo_params)  # speed up subsequent calculations

# plot_densities(density(phi0), density(phi1))


# # # =============================================================================
# # # Visualize spin modes
# # # =============================================================================
# spinz_operator = np.kron(sigma_z, sigma_0)

# spin_density = kwant.operator.Density(fsyst.leads[0], spinz_operator)
# spin_density = spin_density.bind(params=topo_params)

# # plot_densities(spin_density(phi0), spin_density(phi1), title='Spin')

# # =============================================================================
# # Calculate current
# # =============================================================================
# S = kwant.smatrix(fsyst, energy=0, params=topo_params)

# t_10 = S.submatrix(1, 0)  # transmission block, from lead 0 to lead 1

# print(t_10)  
# print(np.trace(t_10.T.conjugate() @ t_10))  # manually calculate transmission
# print(S.transmission(1, 0))  # Kwant has an easy method for this
# def prepare_axes(title):
#     _, (ax0, ax1) = plt.subplots(1, 2)
#     for ax in (ax0, ax1):
#         ax.set_xlabel('x')
#         ax.set_ylabel('y')
#     ax0.set_title(title + ' from left lead')
#     ax1.set_title(title + ' from right lead')
#     return (ax0, ax1)


# scattering_states = kwant.wave_function(fsyst, energy=0, params=topo_params)

# sl0, sl1 = scattering_states(0)  # from lead 0
# sr0, sr1 = scattering_states(1)  # from lead 1

# spin_density = kwant.operator.Density(fsyst, spinz_operator).bind(params=topo_params)

# (ax0, ax1) = prepare_axes('Spin density')
# kwant.plotter.density(fsyst, spin_density(sl0) + spin_density(sl1), cmap='RdBu_r', ax=ax0)
# kwant.plotter.density(fsyst, spin_density(sr0) + spin_density(sr1), cmap='RdBu_r', ax=ax1)


# current = kwant.operator.Current(fsyst).bind(params=topo_params)

# (ax0, ax1) = prepare_axes('Particle Current');
# kwant.plotter.current(fsyst, current(sl0) , ax=ax0)
# kwant.plotter.current(fsyst, current(sr0), ax=ax1)
# =============================================================================
# conductance by varying chemical potential
# =============================================================================
trivial_param_mu = trivial_params.copy()
topo_param_mu = topo_params.copy()
data_trivial = []
data_topo = []
mus = []
for mu_val in np.linspace(-1,1,100):
    trivial_param_mu.update({'mu_right_lead': -1})
    trivial_param_mu.update({'mu_left_lead': 1})
    trivial_param_mu.update({'mu': mu_val})
    
    topo_param_mu.update({'mu_right_lead': -1})
    topo_param_mu.update({'mu_left_lead': 1})
    topo_param_mu.update({'mu': mu_val})
    # compute the scattering matrix at a given energy
    smatrix_trivial = kwant.smatrix(fsyst,energy=0, params=trivial_param_mu)
    smatrix_topo = kwant.smatrix(fsyst,energy=0, params=topo_param_mu)

    # compute the transmission probability from lead 0 to
    # lead 1
    mus.append(mu_val)
    data_trivial.append(smatrix_trivial.transmission(1, 0))
    data_topo.append(smatrix_topo.transmission(1, 0))
        
plt.figure()
plt.plot(mus, data_trivial, label='trivial')
plt.plot(mus, data_topo, label='topological')
plt.xlabel("chemical potental")
plt.ylabel("conductance [e^2/h]")
plt.legend()
plt.show()
# # =============================================================================
# # conductance by varying energies
# # =============================================================================
# energies = []
# data = []
# for energy in np.linspace(0,1.5,100):

#     # compute the scattering matrix at a given energy
#     smatrix = kwant.smatrix(fsyst,energy=energy, params=topo_params)

#     # compute the transmission probability from lead 0 to
#     # lead 1
#     energies.append(energy)
#     data.append(smatrix.transmission(1, 0))
    
# plt.figure()
# plt.plot(energies, data)
# plt.xlabel("energy [t]")
# plt.ylabel("conductance [e^2/h]")
# plt.show()