from sympy import *
from sympy.physics.quantum import *
import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

Msn, Mte, kx, ky, kz, tnn, tsn, tte, lambdasn, lambdate, kx2, ky2, kze = symbols('Msn Mte kx ky kz tnn tsn tte lambdasn lambdate kx2 ky2 kz2')

#Pauli Matrices
sigma_0 = eye(2)
sigma_x = Matrix([[0, 1], [1, 0]])
sigma_y = Matrix([[0, -1j], [1j, 0]])
sigma_z = Matrix([[1, 0], [0, -1]])

Honsite = TensorProduct(Matrix([[Msn, 0],[0, Mte]]), eye(3))
Hnn = 2*tnn*TensorProduct(sigma_x,diag(cos(kx), cos(ky), cos(kz)))

Mnnn = Matrix([[cos(kx)*(cos(ky)+cos(kz)), -sin(kx)*sin(ky), -sin(kx)*sin(kz)],[-sin(kx)*sin(ky), cos(ky)*(cos(kx)+cos(kz)), -sin(ky)*sin(kz)],[-sin(kx)*sin(kz), -sin(ky)*sin(kz),cos(kz)*(cos(kx)+cos(ky))]])
Hnnn = 2*TensorProduct(diag(tsn, tte),Mnnn)

Mso = TensorProduct(Matrix([[0,1,0],[-1,0,0],[0,0,0]]), sigma_z)+TensorProduct(Matrix([[0,0,-1],[0,0,0],[1,0,0]]),sigma_y)+TensorProduct(Matrix([[0,0,0],[0,0,1],[0,-1,0]]),sigma_x)
Hso = -1j/2*TensorProduct(diag(lambdasn, lambdate),Mso)

Htotal = TensorProduct(Honsite+Hnn+Hnnn,sigma_0)+Hso



#for ii in range(int(np.sqrt(len((Heval))))):
#    for jj in range(int(np.sqrt(len((Heval))))):
#        Heval[ii,jj]=Heval[ii,jj].series(kx,x0=1.5,n=4).removeO().series(ky,x0=1.5,n=4).removeO().series(kz,x0=1.5,n=4).removeO()

#print(Heval)
# for kxx in np.linspace(0,pi,20):
#     Energyval = np.array(list(Heval.subs([(kx, kxx),(ky,pi),(kz,pi)]).eigenvals().keys()))
#     Energy = []
#     for item in Energyval:
#         Energy.append(float(re(item)))
        
#     plt.scatter((np.ones(len(Energy))*kxx).tolist(),Energy)
for x in np.linspace(0,1,21):
    Heval = Htotal.subs([(Msn,(1-x)*-1.65+x*-2.27),(Mte,(1-x)*1.65+x*2.27),(tnn,0.9),(tsn,-0.5),(tte,0.5),(lambdasn,(1-x)*0.7+x*0.5),(lambdate,(1-x)*0.7+x*1.5)])
    pi = 314159265358979/100000000000000
    for kxx in np.linspace(0,2*pi,51):
        Energyval = np.array(Heval.subs([(kx, kxx),(ky,kxx),(kz,kxx)]), dtype = complex)
        for ii in range(len(Energyval)):
            for jj in range(len(Energyval)):
                Energyval[ii,jj] = complex(Energyval[ii,jj])
        Energy = np.linalg.eigvals(Energyval)
    
        plt.scatter(kxx*np.ones(12),Energy)
    plt.show()