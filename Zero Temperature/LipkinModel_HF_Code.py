import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fminbound  
import argparse



class Lipkin_Model_HF:
    def __init__(self,e,N,V):

        self.e = e
        self.N = N
        self.V = V


        if abs(self.e/(self.V*(self.N-1)))<=1:
            self.X1 = True
        else:
            self.X1 = False



#Analytic Solution to the Energy
    def Analytic_Sol(self):

        if self.X1==True:
            alpha = 1/2*np.arccos(self.e/(self.V*(self.N-1)))
        else:
            alpha = 0

        E = -self.e*self.N/2*np.cos(2*alpha)-self.V*self.N/4*np.sin(2*alpha)**2*(self.N-1)
        print("Analytic energy =",E)

        return E


### Numerical solution

    def Hartree_Fock_Eqs(self,d0):

        #print(p)
        d1 = np.sqrt(1-d0**2)
        Q = self.V*(self.N-1)*d0*d1/self.e

        e0 = -self.e*np.sqrt(1/4+Q**2)
        
        return (abs(-1/2*d0-Q*d1-(e0/(self.e)*d0)))


    def Numerical_Solver(self):

        if self.X1==True:
            result = fminbound(self.Hartree_Fock_Eqs, 0,1,full_output=True)
            print("Numerical solver converged:",bool(1-result[2]))
            D0 = result[0]
            print("D0 =",D0,"D1 =",-np.sqrt(1-D0))
            
            print("Should be zero:",self.Hartree_Fock_Eqs(D0))
            D1 = -np.sqrt(1-D0**2)

        else:
            D0 = 1
            D1 = 0
        E_num = -self.e*self.N/2*(D0**2-D1**2)-self.V*self.N*(self.N-1)*((D0**2)*(D1**2))

        print("Numerical E =",E_num)



        return E_num
    def delta(self,a,b):
        return int(a==b)


    def MatrixElement(self,m,n):
        N = self.N

        return np.sqrt((N/2-m)*(N/2+m+1)*(N/2+n)*(N/2-n+1))





    def Diagonalization(self):
        ms = [-self.N/2+i for i in range(N+1)]

        H = [[self.e*m*self.delta(n,m)-1/2*self.V*(self.MatrixElement(m,n)*self.delta(n-1,m+1)
                                                +self.MatrixElement(n,m)*self.delta(m-1,n+1)) 
            for m in ms] for n in ms]

        E = np.linalg.eigvalsh(H).min()

        print("Diagonalized energy =",E)

        return E
        

if __name__ == "__main__":
   N = 4
   eps = np.linspace(0,9.5,30)+.5
   V = 10/9

   lhf = Lipkin_Model_HF(eps[0],N,V)
   print(lhf.Numerical_Solver())

   lhf.Diagonalization()


   analyticEnergies1 = []
   numericalEnergies1 = []
   diagonalizedEnergies1 = []
   alphas = []
   
   
   for ep in eps:
   
      lhf = Lipkin_Model_HF(ep,N,V)
      anaE = lhf.Analytic_Sol()
      numE = lhf.Numerical_Solver() 
      diagE = lhf.Diagonalization()
   
   
   
      analyticEnergies1.append(anaE)
      numericalEnergies1.append(numE)
      diagonalizedEnergies1.append(diagE)
   
   
      print("###")
   
   
   plt.scatter(eps,analyticEnergies1,c='r',label="Analytic solution")
   plt.scatter(eps,numericalEnergies1,c='b',marker='^',label="Numerical solution")
   plt.scatter(eps,diagonalizedEnergies1,c='g',marker='*',label="Diagoanlized solution")
   plt.title("$E^{HF}_0$ vs $\\epsilon$ for $\\chi>1$")
   plt.xlabel("Single particle energy ($\\epsilon$)")
   plt.ylabel("Hartree-Fock Energy ($E^{HF}_0$)")
   plt.legend()
   plt.show()






