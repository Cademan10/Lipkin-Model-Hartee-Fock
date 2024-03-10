import numpy as np
from scipy.optimize import fsolve 
import matplotlib.pyplot as plt 

#imports zero temperature code 
import sys
sys.path.append(r'/Users/cader/Desktop/Yale Research/Lipkin Model/Zero Temperature')

from LipkinModel_HF_Code import Lipkin_Model_HF

#LipkinModel_HF_Code.Lipkin_Model_HF(0.5,4,10/9)

class FT_Lipkin_Model_HF:  
	def __init__(self,e,N,V,T,a):

		self.e = e
		self.N = N
		self.V = V
		self.T = T
		self.alpha = a	##convergence condition







	def Alpha(self,e0,e1):
		if self.T == 0:
			return 0
		print(1/(np.exp(e0/self.T)+1)+(1/(np.exp(e1/self.T)+1)))
		al = fsolve(lambda a: 1/(np.exp(e0/self.T+a)+1)+(1/(np.exp(e1/self.T+a)+1))-1,0)
		
		return al[0]


	def HartreeFock(self):
		
		Dm0 = np.cos(np.pi/4)
		Dp0 = -np.sin(np.pi/4)
		Dm1 = np.sin(np.pi/4)
		Dp1 = np.cos(np.pi/4)
		f0 = 1
		f1 = 0
		alpha = 1

		while alpha>self.alpha:
			print("loop iterating...")
			HFmatrix =[[-self.e/2,-self.V*(Dp0*np.conjugate(Dm0)*f0+Dp1*np.conjugate(Dm1)*f1)*(N-1)],\
			[-self.V*(np.conjugate(Dp0)*Dm0*f0+np.conjugate(Dp1)*Dm1*f1)*(N-1),self.e/2]]
			print(HFmatrix)
			eps,D = np.linalg.eig(HFmatrix)
			Dm0_old,Dp0_old,Dm1_old,Dp1_old = Dm0, Dp0, Dm1, Dp1

			if eps[0]<eps[1]:
				e0 = np.real(eps[0])
				e1 = np.real(eps[1])
				Dm0,Dp0 = D[:,0]
				D1m,D1p = D[:,1]

			elif eps[1]<eps[0]:
				e0 = np.real(eps[1])
				e1 = np.real(eps[0])
				Dm0,Dp0 = D[:,1]
				D1m,D1p = D[:,0]
			a = self.Alpha(e0,e1)
			

			if self.T == 0:
				f0=1
				f1 = 0
			
			else:
				f0 = 1/(1+np.exp(e0/self.T+a))
				f1 = 1/(1+np.exp(e1/self.T+a))

			alpha = np.sqrt(np.abs((Dm0_old-Dm0))**2+np.abs((Dm1_old-Dm1))**2+np.abs((Dp0_old-Dp0))**2+np.abs((Dp1_old-Dp1))**2)
		energy = self.N*self.e/2*((np.abs(Dp0)**2-np.abs(Dm0)**2)*f0+(np.abs(Dp1)**2-np.abs(Dm1)**2)*f1)\
		-1/2*self.V*self.N*(self.N-1)*((Dp0*np.conjugate(Dm0)*f0+Dp1*np.conjugate(Dm1)*f1)**2+(Dm0*np.conjugate(Dp0)*f0+Dm1*np.conjugate(Dp1)*f1)**2)

		print("####")
		print("T = ",self.T)
		print(np.real(energy))
		print("#####")
		return np.real(energy)

		


	


#Ns=np.arange(2,10,2)
Ns = [8]
spe = np.linspace(0,9.5,30)+0.5
V = 0.1
Ts=np.linspace(0,20,30)+.001
#print("single particle energy = ",spe[0])
#print(FT_Lipkin_Model_HF(spe[0],N,V,10,.001).HartreeFock())
#print(Lipkin_Model_HF(spe[0],N,V).Numerical_Solver())
print("Ns = ",Ns)

for N in Ns:


	energies = []
	for i in range(len(Ts)):

		lhf = FT_Lipkin_Model_HF(spe[0],N,V,Ts[i],.001)
		energies.append(np.real(lhf.HartreeFock()))

	plt.scatter(Ts,energies,label="N = "+str(N))



	zeroTEnergy = Lipkin_Model_HF(spe[0],N,V).Numerical_Solver()
	plt.plot([Ts[0],Ts[-1]],[zeroTEnergy,zeroTEnergy],"--",zorder=0)

plt.xlabel("$T (k_{B}=1)$")
plt.ylabel("$E_0^{HF}$")
plt.title("$E_{0}^{HF}$ vs $T$ \n $(\\epsilon = $"+str(spe[0])+", $V = $"+str(V)+"$)$")
plt.legend()

plt.savefig("Figures/EnergyVsTemp.png",dpi=700)
plt.show()








energies = []