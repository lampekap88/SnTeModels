import kwant
import scipy.linalg as la
import numpy as np
from matplotlib import pyplot as plt
from numpy import exp, sin, linspace, sqrt, cos
import scipy.sparse.linalg as sla
import math
from model.lcao import L_matrices, lcao_term
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import tinyarray as ta
import plotly
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as io
import time
import pickle
io.renderers.default='browser'

print(kwant.__version__)

def make_system_template(symmetry = None, lattice=None, translations=None, straindirection = []):
    L = L_matrices() #separate script which generates the L matrices
    CFS = np.array([[0,1,1],[1,0,1],[1,1,0]]) #CFS for strain in the (1,1,1) direction

    """Make bulk 6-band model of SnTe from https://arxiv.org/pdf/1202.1003.pdf

    translations: kwant.lattice.TranslationalSymmetry
        Optional different translational unit cell. Should still be a primitive UC.
    lattice: kwant.lattice.Monoatomic
        Optional different lattice to implement strain, should be commensurate with translations.
    """
    
    #pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    sigmas = np.array([sigma_x, sigma_y, sigma_z])
    
    L_dot_S = np.sum([np.kron(sigmas[i], L[i]) for i in range(3)], axis=0)

    
    def onsite(site, m, lam, delta):
        # sum(site.tag) specifies the sublattice on a cubic grid
        a = np.sum(site.tag) % 2
        os = m[a] * np.eye(6)
        # L dot S onsite SoC, Basis: pxu pyu pzu pxd pyd pzd, checked
        spinorb = lam[a] * L_dot_S
        os = os + spinorb
        # crystal field splitting for 111 strain, no strain for #delta = 0
        crystal_field = delta * np.kron(np.eye(2), CFS)
        os += crystal_field
        return os

    def hopping(site1, site2, t,p, strainfactor):
        # which sublattice
        a = np.sum(site1.tag) % 2
        b = np.sum(site2.tag) % 2
        
        #If p=0 and strainfactor = 0 then the transformation matrix is the identity matrix and hence no strain is applied. A model without strain would look like:
         #d = site1.pos-site2.pos
         #dtd = np.kron(np.eye(2), np.outer(d/la.norm(d),d/la.norm(d)))
         #hop = t[a, b] * dtd
        
        # Use the appropriate hopping depending on sublattices
        transformationmatrix = ta.matrix([[1,0,0],[0, (1-(p**2)/2),(np.sqrt((p**2-(1/4)*p**4)))],[0,(np.sqrt((p**2-(1/4)*p**4))), (1-(p**2)/2)]])
        #transformationmatrix = ta.matrix([[(1-(p**2)/2), (np.sqrt((1/2)*(p**2-(1/4)*p**4))), (np.sqrt((1/2)*(p**2-(1/4)*p**4)))],[(np.sqrt((1/2)*(p**2-(1/4)*p**4))), (1-(p**2)/2),(np.sqrt((1/2)*(p**2-(1/4)*p**4)))],[(np.sqrt((1/2)*(p**2-(1/4)*p**4))),(np.sqrt((1/2)*(p**2-(1/4)*p**4))), (1-(p**2)/2)]])
        dold = site1.pos - site2.pos
        
        newpos1 = ta.dot(transformationmatrix, site1.pos)
        newpos2 = ta.dot(transformationmatrix, site2.pos)
        d = newpos1 - newpos2        

        dtd = np.kron(np.eye(2), np.outer(d/la.norm(d),d/la.norm(d)))
        
        hop = t[a, b] * dtd *(1+strainfactor*(1-la.norm(d)/la.norm(dold)))
        return hop

    if lattice is None:
        # Cubic rocksalt structure with FCC symmetry, 3 dimensions with 6 orbitals on each site
        lattice = kwant.lattice.general(np.eye(3), norbs=6)
    if translations is None:
        # Default translation vectors of FCC structure
        translations = kwant.lattice.TranslationalSymmetry([1, 1, 0], [1, 0, 1], [0, 1, 1])
    syst = kwant.Builder(symmetry=translations)

    # Populate the builder using the cubic lattice sites
    # Two sublattices
    syst[lattice(0, 0, 0)] = onsite
    syst[lattice(0, 0, 1)] = onsite

    # First and second neighbor hoppings
    for d in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        syst[kwant.HoppingKind(d, lattice, lattice)] = hopping
    for d in [(1, 1, 0), (1, 0, 1), (0, 1, 1),
              (1, -1, 0), (1, 0, -1), (0, 1, -1)]:
        syst[kwant.HoppingKind(d, lattice, lattice)] = hopping

    return syst

def build_system(template, symdirection, surface_1, surface_2, plotter = False, identifyer='', path = ''):
    symmetry = kwant.TranslationalSymmetry(symdirection) #Define symmetries
    system = kwant.Builder() #Make system
    
    def lead_shape(site): #define the lead shape (two surfaces were calculated elsewere)
        (x,y,z) = site.pos
        return surface_1(x,y,z) and surface_2(x,y,z)
    
    def shape(site): #Define the shape of the "bulk" where the wave function can be calculated
        (x,y,z) = site.pos
        return 0<=symdirection[0]*x+symdirection[1]*y+symdirection[2]*z<=2 and lead_shape(site)
        
    #Fill the system with the tight binding model
    system.fill(template, shape, start=(0, 0, 0));
    
    #Attach leads
    system_lead = kwant.Builder(symmetry=symmetry)
    system_lead.fill(template, lead_shape, start=(0,0,0))
    
    system.attach_lead(system_lead)
    system.attach_lead(system_lead.reversed())
    
    system = system.finalized()
    system_lead =system_lead.finalized()
    
    #Plots the structure. This code will likely only work for SnTe and is hard to change. It creates a separate lattice in a different which can be plotted with 2 different colors for each atom.
    if plotter == True:
        ### make system with sublattice with separate template so you can get colors
        PlotLattice = kwant.lattice.general([(1, 1, 0), (1, 0, 1), (0, 1, 1)],
                                           [(0, 0, 0), (0, 0, 1)])
        a, b = PlotLattice.sublattices
        
        
        def lead_shape_plot(pos):
            x, y, z = pos
            return surface_1(x,y,z) and surface_2(x,y,z)
        
        def shape_plot(pos):
            x, y, z = pos
            return -2<=symdirection[0]*x+symdirection[1]*y+symdirection[2]*z<=2 and lead_shape_plot(pos)
        
        PlotSystem = kwant.Builder()
        PlotSystem[PlotLattice.shape(shape_plot, (0,0,0))] = 0
        PlotSystem[PlotLattice.neighbors()]=-1
        
        PlotLead = kwant.Builder(symmetry=symmetry)
        PlotLead[PlotLattice.shape(lead_shape_plot, (0,0,0))] = 0
        PlotLead[PlotLattice.neighbors()]=-1
        
        PlotSystem.attach_lead(PlotLead)
        PlotSystem.attach_lead(PlotLead.reversed())
        
        fig = plt.figure()
        ax = Axes3D(fig)
        kwant.plot(PlotSystem, show=False, ax=ax)
        legend_elements = [Line2D([0],[0],marker='o', color='w', label='Sn', markerfacecolor='royalblue', markersize = 15),Line2D([0],[0],marker='o', color='w', label='Te', markerfacecolor='orange', markersize = 15)]
        ax.legend(handles=legend_elements, loc='upper right')
        plt.title("Geometric representation of the tight binding model")
        plt.savefig(path+'/SnTe'+identifyer+'Geometry.png')
        plt.show()
        
        #kwant.plotter.set_engine('plotly')
        #kwant.plot(PlotSystem)
        #kwant.plotter.get_engine()
        #kwant.plotter.set_engine('matplotlib')
        
    return system, system_lead

def create_surfaces(surface1, surface2, s1, s2):
    #Creates 2 surfaces from 2 normal vectors of the surfaces. The bulk will run from 0 to s1/s2
    normsurf10 = surface1[0]/np.sqrt(surface1[0]**2+surface1[1]**2+surface1[2]**2)
    normsurf11 = surface1[1]/np.sqrt(surface1[0]**2+surface1[1]**2+surface1[2]**2)
    normsurf12 = surface1[2]/np.sqrt(surface1[0]**2+surface1[1]**2+surface1[2]**2)
    
    normsurf20 = surface2[0]/np.sqrt(surface2[0]**2+surface2[1]**2+surface2[2]**2)
    normsurf21 = surface2[1]/np.sqrt(surface2[0]**2+surface2[1]**2+surface2[2]**2)
    normsurf22 = surface2[2]/np.sqrt(surface2[0]**2+surface2[1]**2+surface2[2]**2)
    
    
    def surface_1(x,y,z):
        return 0<=normsurf10*x+normsurf11*y+normsurf12*z<s1
    
    def surface_2(x,y,z):
        return 0<=normsurf20*x+normsurf21*y+normsurf22*z<s2
    
    return surface_1, surface_2

def plot_spectrum(lead, params, momenta, Energy, identifyer='', path = ''):
    #Plots band spectrum of the lead. It also saves all the bands in a file so it can later be plotted again.
    bands = kwant.physics.Bands(lead, params=params)
    energies = [bands(k) for k in momenta]
    
    f = open(path+'/SnTe'+identifyer+'bands','wb')
    pickle.dump(bands,f)
    f.close()
    
    plt.plot(momenta, energies)
    plt.grid()
    plt.xlabel("Momentum ($a^{-1}$)")
    plt.ylabel("Energy (Ev)")
    plt.ylim(-0.4,0.1)
    plt.axhline(y=Energy, color='black', ls='dashed', label=f'$Energy$={Energy}')
    
    plt.title('Bandspectrum')
    plt.savefig(path+'/SnTe'+identifyer+'bandstructure.png')
    plt.legend()
    plt.show()
    
    

def plot_wavefunction(syst, params, symmetry, layer=0, Energy = 0, a=1, identifyer = '', path = '', Multilayer = False):
    
    #Shows the wave function of the bulk at a given energy. It also shows the all spins (but only for the current basis used)
    wave=kwant.wave_function(syst, energy=Energy, params=params)
    print('wave')
    print(wave)

    Sites=syst.sites
    
    tau_x = np.array([[0,1], [1, 0]])
    tau_y = np.array([[0, -1j], [1j, 0]])
    tau_z = np.array([[1,0], [0, -1]])
    
    def cut(site1):
        return symmetry[0]*site1.pos[0]+symmetry[1]*site1.pos[1]+symmetry[2]*site1.pos[2]==layer
    prob_density = kwant.operator.Density(syst,np.eye(6),where=cut)
    print('prob_density')
    print(prob_density)
    spin_densityx = kwant.operator.Density(syst,np.kron(tau_x,np.eye(3)), where=cut)
    spin_densityy = kwant.operator.Density(syst,np.kron(tau_y,np.eye(3)), where=cut)
    spin_densityz = kwant.operator.Density(syst,np.kron(tau_z,np.eye(3)), where=cut)
    wf = sum(prob_density(psi) for psi in wave(0))
    print('wave(0)')
    print(wave(0))
    print('wf')
    print(wf)
    rho_sx = sum(spin_densityx(psi) for psi in wave(0))
    rho_sy = sum(spin_densityy(psi) for psi in wave(0))
    rho_sz = sum(spin_densityz(psi) for psi in wave(0))
    lat2=kwant.lattice.square(norbs=1,a = a)
    sys2=kwant.Builder()
    
    #Make sure there is always a valid projection
    if symmetry[0]!=0:
        a, b= 1, 2
    elif symmetry[1]!=0:
        a, b = 0, 2
    else:
        a, b = 0, 1
    
    sys2[(lat2(site.pos[a],site.pos[b]) for site in Sites if symmetry[0]*site.pos[0]+symmetry[1]*site.pos[1]+symmetry[2]*site.pos[2]==layer)]=0
    sys2[lat2.neighbors()]=-1
    sys2f=sys2.finalized()
    if type(wf) == int:
        Warning('Energy value is in gap')
        
    else:
        #Plots all wave functions again but this time a layer further.
        kwant.plotter.map(sys2f, (wf), show=False, vmin=0)
        plt.title('Density plot of the cross section of the ('+ '%01d' %(symmetry[0])+ '%01d' %(symmetry[1])+ '%01d' %(symmetry[2])+') plane')
        plt.savefig(path+'/SnTe'+identifyer+'densityplot.png')
        plt.show()
        
        kwant.plotter.map(sys2f, rho_sx, show = False, cmap='bwr', vmin=-0.3, vmax = 0.3)
        plt.title('Spin x density plot of the cross section of the ('+ '%01d' %(symmetry[0])+ '%01d' %(symmetry[1])+ '%01d' %(symmetry[2])+') plane')
        plt.savefig(path+'/SnTe'+identifyer+'spindensityplotx.png')
        plt.show()
        
        kwant.plotter.map(sys2f, rho_sy, show = False, cmap='bwr', vmin=-0.3, vmax = 0.3)
        plt.title('Spin y density plot of the cross section of the ('+ '%01d' %(symmetry[0])+ '%01d' %(symmetry[1])+ '%01d' %(symmetry[2])+') plane')
        plt.savefig(path+'/SnTe'+identifyer+'spindensityploty.png')
        plt.show()
        
        kwant.plotter.map(sys2f, rho_sz, show = False, cmap='bwr', vmin=-0.3, vmax = 0.3)
        plt.title('Spin z density plot of the cross section of the ('+ '%01d' %(symmetry[0])+ '%01d' %(symmetry[1])+ '%01d' %(symmetry[2])+') plane')
        plt.savefig(path+'/SnTe'+identifyer+'spindensityplotz.png')
        plt.show()
        
        if Multilayer == True:
            def cut2(site1):
                return symmetry[0]*site1.pos[0]+symmetry[1]*site1.pos[1]+symmetry[2]*site1.pos[2]==(layer+2)
            prob_density2 = kwant.operator.Density(syst,np.eye(6),where=cut2)
            spin_densityx2 = kwant.operator.Density(syst,np.kron(tau_x,np.eye(3)), where=cut2)
            spin_densityy2 = kwant.operator.Density(syst,np.kron(tau_y,np.eye(3)), where=cut2)
            spin_densityz2 = kwant.operator.Density(syst,np.kron(tau_z,np.eye(3)), where=cut2)
            wf2 = sum(prob_density2(psi) for psi in wave(0))
            print(wf2)
            
            rho_sx2 = sum(spin_densityx2(psi) for psi in wave(0)) 
            rho_sy2 = sum(spin_densityy2(psi) for psi in wave(0)) 
            rho_sz2 = sum(spin_densityz2(psi) for psi in wave(0)) 
            
            lat3=kwant.lattice.square(norbs=1,a = a)
            sys3=kwant.Builder()
            sys3[(lat3(site.pos[a],site.pos[b]) for site in Sites if symmetry[0]*site.pos[0]+symmetry[1]*site.pos[1]+symmetry[2]*site.pos[2]==(layer+2))]=0
            sys3[lat3.neighbors()]=-1
            sys3f=sys3.finalized()

            kwant.plotter.map(sys3f, (wf2), show=False, vmin=0)
            plt.title('Density plot of the cross section of the ('+ '%01d' %(symmetry[0])+ '%01d' %(symmetry[1])+ '%01d' %(symmetry[2])+') plane')
            plt.savefig(path+'/SnTe'+identifyer+'densityplot2.png')
            plt.show()
            
            kwant.plotter.map(sys3f, rho_sx2, show = False, cmap='bwr', vmin=-0.3, vmax = 0.3)
            plt.title('Spin x density plot of the cross section of the ('+ '%01d' %(symmetry[0])+ '%01d' %(symmetry[1])+ '%01d' %(symmetry[2])+') plane')
            plt.savefig(path+'/SnTe'+identifyer+'spindensityplotx2.png', vmin=-1, vmax = 1)
            plt.show()
            
            kwant.plotter.map(sys3f, rho_sy2, show = False, cmap='bwr', vmin=-0.3, vmax = 0.3)
            plt.title('Spin y density plot of the cross section of the ('+ '%01d' %(symmetry[0])+ '%01d' %(symmetry[1])+ '%01d' %(symmetry[2])+') plane')
            plt.savefig(path+'/SnTe'+identifyer+'spindensityploty2.png')
            plt.show()
            
            kwant.plotter.map(sys3f, rho_sz2, show = False, cmap='bwr', vmin=-0.3, vmax = 0.3)
            plt.title('Spin z density plot of the cross section of the ('+ '%01d' %(symmetry[0])+ '%01d' %(symmetry[1])+ '%01d' %(symmetry[2])+') plane')
            plt.savefig(path+'/SnTe'+identifyer+'spindensityplotz2.png')
            plt.show()
        return np.log(wf)
    
def plot_current(syst, params, symmetry, layer=0, Energy=0, identifyer = '', path = ''):
    #Function is currently not working
    tau_x = np.array([[0,1], [1, 0]])
    tau_y = np.array([[0, -1j], [1j, 0]])
    tau_z = np.array([[1,0], [0, -1]])
    
    wf = kwant.wave_function(syst, energy = Energy, params = params)
    psi = wf(0)
    
    if np.size(psi)==0:
        Warning('Energy value is in gap')
        
    else:
        
        J0 = kwant.operator.Current(syst, onsite = np.eye(6))
        Jx = kwant.operator.Current(syst, onsite = np.kron(tau_x,np.eye(3)))
        Jy = kwant.operator.Current(syst, onsite = np.kron(tau_y,np.eye(3)))
        Jz = kwant.operator.Current(syst, onsite = np.kron(tau_z,np.eye(3)))
        
        current0 = J0(psi)
        currentx = Jx(psi)
        currenty = Jy(psi)
        currentz = Jz(psi)
    
        kwant.plotter.current(syst, current0, show = False)
        plt.title('Current0')
        plt.savefig(path+'/SnTe'+identifyer+'currentplot0.png')
        plt.show()
    
        kwant.plotter.current(syst, currentx, show = False)
        plt.title('Currentx')
        plt.savefig(path+'/SnTe'+identifyer+'currentplotx.png')
        plt.show()
    
        kwant.plotter.current(syst, currenty, show = False)
        plt.title('Currenty')
        plt.savefig(path+'/SnTe'+identifyer+'currentploty.png')
        plt.show()
    
        kwant.plotter.current(syst, currentz, show = False)
        plt.title('Currentz')
        plt.savefig(path+'/SnTe'+identifyer+'currentplotz.png')
        plt.show()


def Calculate_pen_depth(wf, s1, s2, identifyer, path):
    #Not used
    if len(wf) != s1*s2:
        return 0
    else:
        rwf = wf.reshape((s1,s2))
        fig, ax = plt.subplots()
        ax.plot(range(s1),rwf[math.ceil(s2/2),:])
        plt.grid()
        plt.xlabel("Site")
        plt.ylabel("Density")
        plt.title("Log plot of the Density")
        plt.savefig(path+'/SnTe'+identifyer+'penetrationdepth.png')
        plt.show()
        pen,b = np.polyfit([0,1,2,3,4],rwf[math.ceil(s2/2),0:5],1)
        print('---')
        print('The penetration depth is '+str(-1/pen))
        print('---')
        
        return -1/pen
 
def Save_parameters(params,identifyer='', path=''):
    #saves all parameters used in a text file
    f = open(path+'/SnTe'+identifyer+'.txt',"w+")
    for i in range(len(params)):
        f.write(params[i]+'\n')
    f.close
    
def plot_crosssection(surface_1,surface_2,symmetry,path, show = False, identifyer = ''):
    #Plots the lead in an interactive plot
    x1 = range(0,21,1)  
    
    data_x1 = []
    data_y1 = []
    data_z1 = []
    
    colordata =[]
    
    
    for x in x1:
        for y in x1:
            for z in x1:
                if (surface_1(x,y,z) & surface_2(x,y,z)).all():
                    data_x1.append(x)
                    data_y1.append(y)
                    data_z1.append(z)
                    if (x+y+z) %2 == 1:
                        colordata.append(0)
                    else:
                        colordata.append(1)
                    
                
    fig = go.Figure(data=[go.Scatter3d(x=data_x1, y=data_y1, z=data_z1, mode = 'markers', marker=dict(size = 8, color=colordata, opacity = 0.5, colorscale = 'Jet'))])
    if show:    
        fig.show()
    
    fig.write_html(path+'/SnTe'+identifyer+'interactivegeometry.html')

    
def main(s1 = 12,s2= 12,a=1, Energy=-0.0, symmetry = (2,0,0), surface1 = (0,0,1), surface2 = (0, 1, 0), delta = 0, p=0, strainfactor=0, comp=1, newpath = '/Figure/BandStructurePbSnTe'):   #comp=1 then SnTe, PbTe values for lambda have been estimated from Max calcs
    SnTe_6band_params = {'t': np.array([[-0.5,  0.9],[ 0.9,  0.5]]), 'm': comp*np.array([-1.65,  1.65])+(1-comp)*np.array([-2.27,  2.27]), 'lam': comp*np.array([0.7, 0.7]) + (1-comp)*np.array([0.5, 1.5]), 'p': p, 'strainfactor': strainfactor, 'delta': delta}
    
    #Creates identifier such that all plots will be saved when engaging in a for loop.
    identifyer = '%02d' %a + '%02d' %(s1) + '%02d' %(s2) + '%02d' %(Energy*1000) + '%01d' %(symmetry[0])+ '%01d' %(symmetry[1])+ '%01d' %(symmetry[2])+ '%01d' %(surface1[0])+ '%01d' %(surface1[1])+ '%01d' %(surface1[2])+ '%01d' %(surface2[0])+ '%01d' %(surface2[1])+ '%01d' %(surface2[2]) + '%02d' %(delta*100)+ '%02d' %(p*100) + '%02d' %(strainfactor*100) + '%02d' %(comp*100)
    
    #Make folder
    path = os.path.dirname(__file__)
    totalpath = path+newpath
    if not os.path.exists(totalpath):
        os.makedirs(totalpath)
    if not os.path.exists(totalpath+'/Geometry'):
        os.makedirs(totalpath+'/Geometry')
    if not os.path.exists(totalpath+'/InteractiveGeometry'):
        os.makedirs(totalpath+'/InteractiveGeometry')
    if not os.path.exists(totalpath+'/BandStructure'):
        os.makedirs(totalpath+'/BandStructure')
    if not os.path.exists(totalpath+'/Density'):
        os.makedirs(totalpath+'/Density')
    if not os.path.exists(totalpath+'/Pendepth'):
        os.makedirs(totalpath+'/Pendepth')
    if not os.path.exists(totalpath+'/Parameters'):
        os.makedirs(totalpath+'/Parameters')
    if not os.path.exists(totalpath+'/Current'):
        os.makedirs(totalpath+'/Current')
    
    SnTe_template = make_system_template(straindirection = (1,1,0))
    
    params = SnTe_6band_params
    
    surface_1, surface_2 = create_surfaces(surface1, surface2, s1, s2)
    
    plot_crosssection(surface_1 = surface_1,surface_2 = surface_2,symmetry = symmetry, path = totalpath+'/InteractiveGeometry', show=False, identifyer = identifyer)
    
    SnTe, SnTe_lead = build_system(SnTe_template, symmetry, surface_1, surface_2, plotter = False, identifyer = identifyer, path = totalpath+'/Geometry')
    
    plot_spectrum(SnTe_lead, params, np.linspace(-math.pi/a,0, 101), Energy=Energy, identifyer = identifyer, path = totalpath+'/BandStructure')
    
    plot_wavefunction(SnTe, params, symmetry, Energy = Energy, identifyer = identifyer, path = totalpath+'/Density', Multilayer = True)
    
    pen=-1
    
    #plot_current(syst = SnTe, params=params, Energy = Energy, identifyer = identifyer, path = totalpath+'/Current', symmetry = symmetry)
    #pen = Calculate_pen_depth(wf, s1, s2, identifyer, path = totalpath+'/Pendepth')
    
    #Save all important parameters and results
    parameters = ('a='+str(float(a)),'s1='+str(int(s1)),'s2='+str(int(s2)),'Energy='+str(float(Energy)),'symmetry=('+str(int(symmetry[0]))+str(int(symmetry[1]))+str(int(symmetry[2]))+')','surface_1=('+str(int(surface1[0]))+str(int(surface1[1]))+str(int(surface1[2]))+')','surface_2=('+str(int(surface2[0]))+str(int(surface2[1]))+str(int(surface2[2]))+')','pen='+str(float(1/pen)),'delta='+str(float(delta)),'p='+str(float(p)),'strainfactor='+str(float(strainfactor)),'comp='+str(float(comp)))
    Save_parameters(parameters, identifyer=identifyer, path = totalpath+'/Parameters')
    
    

if __name__=='__main__':
    start = time.time()
    main(s1 = 7, s2 = 7, delta = 0, Energy = -0.087, comp = 1)
    end = time.time()
    print('Time')
    print(end-start)

    