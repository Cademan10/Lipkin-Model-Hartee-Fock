import numpy as np
from numpy import cos,sin
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import fminbound  


class Lipkin_Model_HF:
    def __init__(self,x,N,e):
        self.N = N
        self.x = x
        self.e = e

        self.V = e*x/(N-1)

        if x>1:
            self.alpha = 1/2*np.arccos(self.e/(self.V*(self.N-1)))
        else:
            self.alpha = 0
 

#Analytic Solution to the Energy
    def Analytic_Sol(self):
        print("###")

        E = -self.e*self.N/2*np.cos(2*self.alpha)-self.V*self.N/4*np.sin(2*self.alpha)**2*(self.N-1)
        print("Analytic energy =",E)



        return E


### Numerical solution

    def Hartree_Fock_Eqs(self,d0):

        #print(p)
        d1 = np.sqrt(1-d0**2)
        Q = self.V*(self.N-1)*d0*d1/self.e

        e0 = -self.e*self.N*np.sqrt(1/4+Q**2)
        
        return (abs(-1/2*d0-Q*d1-(e0/(self.e*self.N)*d0)))


    def Numerical_Solver(self):

        print("###")

        if self.x <= 1:
            D0 = 1
            D1 = 0

        else:

            result = fminbound(self.Hartree_Fock_Eqs, 0,1,full_output=True)
            print("Numerical solver converged:",bool(1-result[2]))
            D0 = result[0]
            print("D0 =",D0,"D1 =",-np.sqrt(1-D0))
            
            print("Should be zero:",self.Hartree_Fock_Eqs(D0))
            D1 = -np.sqrt(1-D0**2)


        E_num = -self.e*self.N/2*(D0**2-D1**2)-self.V*self.N*(self.N-1)*((D0**2)*(D1**2))

        print("Numerical E =",E_num)
        



        return (E_num,np.arccos(D0))

    def delta(self,a,b):
        return int(a==b)


    def MatrixElement(self,m,n):
        N = self.N

        return np.sqrt((N/2-m)*(N/2+m+1)*(N/2+n)*(N/2-n+1))
    
    def Diagonalization(self):
        print("###")
        ms = [-self.N/2+i for i in range(N+1)]

        H = [[self.e*m*self.delta(n,m)-1/2*self.V*(self.MatrixElement(m,n)*self.delta(n-1,m+1)
                                                +self.MatrixElement(n,m)*self.delta(m-1,n+1)) 
            for m in ms] for n in ms]
        
        E_diag = np.linalg.eigvalsh(H).min()

        print("Diagonalized energy =",E_diag)
        

        return E_diag


N = 25
e = 20
#xs = [1.8,0.6]
#alphas = np.linspace(-np.pi/2,np.pi/2,50)

xs = np.linspace(0.5,2.5,30)
analyticEnergies1 = []
numericalEnergies1 = []
diagonalizedEnergies1 = []
#analyticEnergies2 = []
#numericals = []


for x in xs:

    
    lhf = Lipkin_Model_HF(x,N,e)
    anaE = lhf.Analytic_Sol()/e
    numE = lhf.Numerical_Solver()[0]/e
    diagE = lhf.Diagonalization()/e

    analyticEnergies1.append(anaE)
    numericalEnergies1.append(numE)
    diagonalizedEnergies1.append(diagE)

    
   

    print("###############")

#numE,alpha = lhf.Numerical_Solver()
#numericals.append([numE,alpha])


plt.scatter(xs,analyticEnergies1,c='r',label="Analytic solution")
plt.scatter(xs,numericalEnergies1,c='b',marker='^',label="Numerical solution")
plt.scatter(xs,diagonalizedEnergies1,c='g',marker='*',label="Diagonalized solution")
#plt.scatter(numericals[0][1]*180/np.pi,numericals[0][0],c='k',marker="*",s=200)
#plt.scatter(-numericals[0][1]*180/np.pi,numericals[0][0],c='k',marker="*",s=200)
#plt.scatter(numericals[1][1]*180/np.pi,numericals[1][0],c='k',marker="*",s=200)
title = "$E_{gs}$ vs $\\chi$ $(N =$ "+str(N)+", $\\epsilon = 3)$"
plt.title(title)
plt.xlabel("$\\chi$")
plt.ylabel("$E_{gs}/\\epsilon$")
plt.legend()

plt.savefig("Figures/GSEvX_N"+str(N)+"_SPE"+str(e)+".png",dpi=500)
#plt.text(55,-13,"$\\chi=1.8$",c='r',fontweight='bold')
#plt.text(25,0,"$\\chi=0.6$",c='b',fontweight='bold')
plt.show()





