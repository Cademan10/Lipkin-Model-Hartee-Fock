import numpy as np
import math
from numpy import sin,cos,log,exp
from scipy.optimize import fsolve 
import matplotlib.pyplot as plt 

#imports zero temperature code 
import sys
sys.path.append(r'/Users/cader/Desktop/Yale Research/Lipkin Model/Zero Temperature/Code')

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
		al = fsolve(lambda a: 1/(exp(e0/self.T+a)+1)+(1/(exp(e1/self.T+a)+1))-1,0)
		
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
			HFmatrix =[[-self.e/2,-self.V*(Dp0*np.conjugate(Dm0)*f0+Dp1*np.conjugate(Dm1)*f1)*(N-1)],\
			[-self.V*(np.conjugate(Dp0)*Dm0*f0+np.conjugate(Dp1)*Dm1*f1)*(N-1),self.e/2]]
			eps,D = np.linalg.eig(HFmatrix)
			Dm0_old,Dp0_old,Dm1_old,Dp1_old = Dm0, Dp0, Dm1, Dp1

			if eps[0]<eps[1]:
				e0 = np.real(eps[0])
				e1 = np.real(eps[1])
				Dm0,Dp0 = D[:,0]
				Dm1,Dp1 = D[:,1]

			elif eps[1]<eps[0]:
				e0 = np.real(eps[1])
				e1 = np.real(eps[0])
				Dm0,Dp0 = D[:,1]
				Dp0,Dp1 = D[:,0]
			a = 0
			

			if self.T == 0:
				f0=1
				f1 = 0
			
			else:
				f0 = 1/(1+np.exp(e0/self.T+a))
				f1 = 1/(1+np.exp(e1/self.T+a))

			alpha = np.sqrt(np.abs((Dm0_old-Dm0))**2+np.abs((Dm1_old-Dm1))**2+np.abs((Dp0_old-Dp0))**2+np.abs((Dp1_old-Dp1))**2)
			
		energy = self.N*self.e/2*((np.abs(Dp0)**2-np.abs(Dm0)**2)*f0+(np.abs(Dp1)**2-np.abs(Dm1)**2)*f1)\
		-1/2*self.V*self.N*(self.N-1)*((Dp0*np.conjugate(Dm0)*f0+Dp1*np.conjugate(Dm1)*f1)**2+(Dm0*np.conjugate(Dp0)*f0+Dm1*np.conjugate(Dp1)*f1)**2)

		return np.real(energy)



class ExactExpressions:  
	def __init__(self,e,N,V,T):
		self.e = e
		self.N = N
		self.V = V
		self.T = T


	def delta(self,m,n):
		return int(m==n)



	def ExactEnergy(self):
		ks = np.arange(0+(N%2)/2,int(N/2)+1)
		PF = 0
		Energy = 0
		for k in ks:

			if k==N/2:
				degenergacy=1
			else:
				degenergacy =  math.factorial(N)**2/(math.factorial(int(N/2-k))**2*math.factorial(int(N/2+k))**2)\
-math.factorial(N)**2/(math.factorial(int(N/2-k-1))**2*math.factorial(int(N/2+k+1))**2)

			energies = self.KSubspace(k)

			Energy+=degenergacy*np.sum(energies*np.exp(-energies/self.T,dtype=np.longdouble),dtype=np.longdouble)

			PF+=degenergacy*np.sum(np.exp(-energies/self.T,dtype=np.longdouble),dtype=np.longdouble)
		return Energy/PF

	def HeatCapacity(self):
		ks = np.arange(0+(N%2)/2,int(N/2)+1)
		PF = 0
		EnergySquared = 0
		for k in ks:

			if k==N/2:
				degenergacy=1
			else:
				degenergacy =  math.factorial(N)**2/(math.factorial(int(N/2-k))**2*math.factorial(int(N/2+k))**2)\
-math.factorial(N)**2/(math.factorial(int(N/2-k-1))**2*math.factorial(int(N/2+k+1))**2)

			energies = self.KSubspace(k)

			EnergySquared+=degenergacy*np.sum((energies**2)*np.exp(-energies/self.T,dtype=np.longdouble),dtype=np.longdouble)

			PF+=degenergacy*np.sum(np.exp(-energies/self.T,dtype=np.longdouble),dtype=np.longdouble)
		#print((EnergySquared/PF-self.ExactEnergy()**2)/self.T**2)
		return (EnergySquared/PF-self.ExactEnergy()**2)/self.T**2



	def KSubspace(self,k):
		
		ms = np.arange(-k,k+1,1)
		H = [[self.e*m*self.delta(m,n)-1/2*self.V*(np.sqrt((k-m)*(k+m+1)*(k+n)*(k-n+1))*self.delta(m+1,n-1)
			+np.sqrt((k-n)*(k+n+1)*(k+m)*(k-m+1))*self.delta(m-1,n+1)) for n in ms] for m in ms]


		return np.linalg.eigvalsh(H)




	


class Free_Energy:
	def __init__(self,e,alphas,N,V,T):
		self.e = e
		self.alphas = alphas
		self.N = N
		self.V = V
		self.T = T


		self.chi = self.V/self.e*(self.N-1)

	def QSquared(self,alpha,f0,f1):
		return self.chi**2*(sin(alpha)*cos(alpha)*f0-cos(alpha)*sin(alpha)*f1)**2

	def Energy(self,e0,alpha):
		f0 = 1/(1+exp(e0/self.T))
		f1 = 1/(1+exp(-e0/self.T))

		E = self.N*self.e/2*((np.abs(sin(alpha))**2-np.abs(cos(alpha))**2)*f0+(np.abs(cos(alpha))**2-np.abs(sin(alpha))**2)*f1)\
		-self.V*self.N*(self.N-1)*((sin(alpha)*cos(alpha)*f0-cos(alpha)*sin(alpha)*f1)**2)
		S = -2*self.N*(f0*log(f0)+f1*log(f1))

		return E-self.T*S

		#return -T*N*(log(1+exp(-e0/self.T))+log(1+exp(e0/self.T)))
	

	def Solver(self):
		energies = []
		for alpha in self.alphas:
			E0 = fsolve(lambda e0:e0+self.e*np.sqrt(1/4+self.QSquared(alpha,1/(1+exp(e0/self.T)),1/(1+exp(-e0/self.T)))),-self.e)[0]
			
			energies.append(self.Energy(E0,alpha)) 
		
		return energies


def HFHeatCapacity(e,N,V,T,dT):
	hf1 = FT_Lipkin_Model_HF(e,N,V,T+dT,.001).HartreeFock()
	hf2 = FT_Lipkin_Model_HF(e,N,V,T-dT,.001).HartreeFock()

	return (hf1-hf2)/(2*dT)



################### Runs the Hartree Fock numerical solver routine ###################

Ns = [10]
spe = np.linspace(1,9.5,30)
V = 0.1
Ts=np.linspace(0,20,30)+.05


for N in Ns:


	energies = []
	exactEnergies = []
	for i in range(len(Ts)):

		lhf = FT_Lipkin_Model_HF(spe[0],N,V,Ts[i],.001)
		lexact = ExactExpressions(spe[0],N,V,Ts[i])
		energies.append(np.real(lhf.HartreeFock()))

		exactEnergies.append(lexact.ExactEnergy())

	plt.scatter(Ts,energies,label="HF Energy (N="+str(N)+")")
	plt.scatter(Ts,exactEnergies,label="Exact Energy (N="+str(N)+")")



	zeroTEnergy = Lipkin_Model_HF(spe[0],N,V).Numerical_Solver()
	zeroTAnaE = Lipkin_Model_HF(spe[0],N,V).Diagonalization()
	plt.plot([Ts[0],Ts[-1]],[zeroTEnergy,zeroTEnergy],"--",zorder=0,label="Numerical T=0 solution (N="+str(N)+")")
	plt.plot([Ts[0],Ts[-1]],[zeroTAnaE,zeroTAnaE],"--",zorder=0,label="Exact T=0 solution (N="+str(N)+")")


plt.xlabel("$T (k_{B}=1)$")
plt.ylabel("$\\langle{E}\\rangle_{HF}$")
plt.title("$\\langle{E}\\rangle_{HF}$ vs $T$ \n $(\\epsilon = $"+str(spe[0])+", $V = $"+str(V)+"$)$")
plt.legend()

plt.savefig("Figures/EnergyVsTemp.png",dpi=700)
plt.show()



################### Runs the routine plotting the free energy as a function of alpha ###################

Vs = np.linspace(0.1,1,10)
spe = 0.5
Ts = np.linspace(0.1,1,100)
N = 20
alphas = np.linspace(-np.pi/2,np.pi/2,1000)

freeEnergies = []
for T in Ts:
	fe = Free_Energy(spe,alphas,N,Vs[0],T)
	energies = fe.Solver()
	freeEnergies.append([T,energies])
	energies = energies-np.min(energies)

#	plt.scatter(180*alphas/np.pi,energies,label="T = "+str(round(T,2)),s=5)
#	plt.legend()
#
#
#plt.xlabel("$\\alpha$")
#plt.ylabel("$F(\\alpha)$")
#plt.title("$F(\\alpha)$ vs $\\alpha$ \n $\\chi = $"+str(Vs[0]/spe*(N-1)))
#plt.savefig("Figures/FreeEnergyVsAlpha.png",dpi=700)
#plt.show()
#



################### Plots F(alpha)_min as a function of T ###################
for i in range(len(freeEnergies)):
	plt.scatter(freeEnergies[i][0],abs(180*alphas[np.max(np.where(freeEnergies[i][1]==np.min(freeEnergies[i][1]))[0])]/np.pi),c="b",s=10)
#
#
#	
plt.xlabel("$T$")
plt.ylabel("$\\alpha_{min}$")
plt.title("$\\alpha_{min}$ vs T, $\\chi = $"+str(Vs[0]/spe*(N-1)))
plt.savefig("Figures/AlphaMinVsT.png",dpi=700)
plt.show()



################### Plot the heat capacity ###################
#Ts = np.linspace(.1,1,80)
#exactCapacities = []
#for T in Ts:
#	plt.scatter(T,HFHeatCapacity(spe,N,V,T,10**(-6)),c="b")
#
#	exactCapacities.append(ExactExpressions(spe,N,V,T).HeatCapacity())
#plt.plot(Ts,exactCapacities,c='r')
#plt.ylabel("$C_{V}$")
#plt.xlabel("$T$")
#plt.title("$C_{V}$ vs $T$ \n N = "+str(N))
#plt.savefig("Figures/HeatCapacity.png",dpi=700)
#plt.show()


