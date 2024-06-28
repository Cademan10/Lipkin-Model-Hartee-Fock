import numpy as np
import math
from numpy import sin,cos,log,exp
from scipy.optimize import fsolve 
import matplotlib.pyplot as plt 

params= {'text.latex.preamble' : r'\usepackage{amsmath}'}
plt.rcParams.update(params)

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


	def HartreeFock(self,entropy=False,partProj=False,projN=False):
		## If entropy == True, this function returns the energy and entropy
		## If entropy == False, returns just the energy

		## If partProj == True, this function returns the particle projected Partition Function
		## If partProj == False, returns just the energy (and entropy if selected)

		Dm0 = np.cos(np.pi/4)
		Dp0 = -np.sin(np.pi/4)
		Dm1 = np.sin(np.pi/4)
		Dp1 = np.cos(np.pi/4)
		f0 = 1
		f1 = 0
		alpha = 1

		while alpha>self.alpha:
			HFmatrix =[[-self.e/2,-self.V*(Dp0*np.conjugate(Dm0)*f0+Dp1*np.conjugate(Dm1)*f1)*(self.N-1)],\
			[-self.V*(np.conjugate(Dp0)*Dm0*f0+np.conjugate(Dp1)*Dm1*f1)*(self.N-1),self.e/2]]
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


		S = -2*self.N*(f0*log(f0)+f1*log(f1))

		if partProj == True:

	

			U0 = -1/2*self.V*self.N*(self.N-1)*((Dp0*np.conjugate(Dm0)*f0+Dp1*np.conjugate(Dm1)*f1)**2+(Dm0*np.conjugate(Dp0)*f0+Dm1*np.conjugate(Dp1)*f1)**2)
			
			#Z_N = 1/(2*self.N)*exp(U0/self.T)*np.sum([exp(-1j*m*np.pi)*(1+exp(-e0/self.T+1j*m*np.pi/self.N))**self.N*(1+exp(e0/self.T+1j*m*np.pi/self.N))**self.N for m in list(range(1,2*self.N+1))])
			log_Z_N = np.real(U0/self.T-log(2*self.N)+log(np.sum([exp(-1j*m*np.pi)*(1+exp(-(e0)/self.T+1j*m*np.pi/self.N))**self.N*(1+exp((e0)/self.T+1j*m*np.pi/self.N))**self.N for m in list(range(1,2*self.N+1))])))
			
			if projN == True:
				N = np.abs(exp(U0/self.T)/2*np.sum([exp(-1j*m*np.pi)*(1+exp(-e0/self.T+1j*m*np.pi/self.N))**self.N*(1+exp(e0/self.T+1j*m*np.pi/self.N))**self.N*(1/(1+np.exp(-1j*np.pi*m/self.N+e0/self.T))+1/(1+np.exp(-1j*np.pi*m/self.N-e0/self.T))) for m in list(range(1,2*self.N+1))])/np.exp(log_Z_N))

				return (log_Z_N,N)
			else:
				return log_Z_N


		if entropy==True:
			return np.real(energy),S
		else:
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
		N = self.N
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


	def PartitonFunc(self):
		N = self.N
		ks = np.arange(0+(N%2)/2,int(N/2)+1)
		PF = 0
		for k in ks:

			if k==N/2:
				degenergacy=1
			else:
				degenergacy =  math.factorial(N)**2/(math.factorial(int(N/2-k))**2*math.factorial(int(N/2+k))**2)\
-math.factorial(N)**2/(math.factorial(int(N/2-k-1))**2*math.factorial(int(N/2+k+1))**2)

			energies = self.KSubspace(k)


			PF+=degenergacy*np.sum(np.exp(-energies/self.T,dtype=np.longdouble),dtype=np.longdouble)
		return PF


	def HeatCapacity(self):
		N = self.N
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


class HF_Heat_Capacity:

	
	def HFHeatCapacity(self,e,N,V,T,dT):
		hf1 = FT_Lipkin_Model_HF(e,N,V,T+dT,.00001).HartreeFock()
		hf2 = FT_Lipkin_Model_HF(e,N,V,T-dT,.00001).HartreeFock()
	
		return (hf1-hf2)/(2*dT)

	def Tc(self,e,N,V,dT):
		T = .05
		sign = -1
		while sign<0:
			T+=.001
			sign = abs(self.HFHeatCapacity(e,N,V,T,dT)-self.HFHeatCapacity(e,N,V,T+.001,dT))-15*abs(self.HFHeatCapacity(e,N,V,T-.001,dT)-self.HFHeatCapacity(e,N,V,T,dT))
#
			if T>=5:
				return 0


		
		heatCaps = []
		Ts = np.linspace(.01,10,200)

		for T in Ts:
			heatCaps.append(self.HFHeatCapacity(e,N,V,T,dT))

		diffs = abs(np.array(heatCaps[1:])-np.array(heatCaps[0:-1]))

		maxDiff = max(diffs)

		if maxDiff>=0.4*max(heatCaps):

			return Ts[np.where(diffs==maxDiff)]
		else:
			return 0




if __name__ == "__main__":

	################### Runs the Hartree Fock numerical solver routine ###################
	
	
	V = 0.1
	spe = [0.5]
	Ts = np.linspace(0.01,20,30)
	N = 10
	 

	
	
	
	energies = []
	exactEnergies = []
	energies_PP = []
	for i in range(len(Ts)):
	
		lhf = FT_Lipkin_Model_HF(spe[0],N,V,Ts[i],.00001)
		lexact = ExactExpressions(spe[0],N,V,Ts[i])
		energies.append(np.real(lhf.HartreeFock()))

		energies_PP.append(Ts[i]**2*(log(lhf.HartreeFock(partProj=True))-log(FT_Lipkin_Model_HF(spe[0],N,V,Ts[i]-10e-7,.00001).HartreeFock(partProj=True)))/10e-7)
	
		exactEnergies.append(lexact.ExactEnergy())


	#
	plt.scatter(Ts,energies,label="HF Energy",c='b')
	plt.scatter(Ts,energies_PP,label="HF Energy (P.P.)",c='g')
	plt.scatter(Ts,exactEnergies,label="Exact Energy",c='orange')
	
	
	
	zeroTEnergy = Lipkin_Model_HF(spe[0],N,V).Numerical_Solver()
	zeroTAnaE = Lipkin_Model_HF(spe[0],N,V).Diagonalization()
	plt.scatter(Ts[0],zeroTEnergy,facecolors='none', edgecolors='r',zorder=100,label="Numerical T=0 solution")
	plt.scatter(Ts[0],zeroTAnaE,facecolors='none', edgecolors='k',zorder=100,label="Exact T=0 solution")
	#
	#
	plt.xlabel("$T (k_{B}=1)$")
	plt.ylabel("$\\langle{E}\\rangle_{HF}$")
	plt.title("$\\langle{E}\\rangle_{HF}$ vs $T$ \n $(\\epsilon = $"+str(spe[0])+", $V = $"+str(V)+"$, N = 10)$")
	plt.legend()
	
	plt.savefig("Figures/Phase Transition/EnergyVsTemp.png",dpi=700)
	plt.show()
	
	
	
	################### Runs the routine plotting the free energy as a function of alpha ###################
	
	#Vs = [0.1]
	#spe = 0.5
	#Ts = np.linspace(0.1,20,300)
	#N = 10
	#alphas = np.linspace(-np.pi/2,np.pi/2,1000)
	#
	#freeEnergies = []
	#for T in Ts:
	#	fe = Free_Energy(spe,alphas,N,Vs[0],T)
	#	energies = fe.Solver()
	#	freeEnergies.append([T,energies])
	#	energies = energies-np.min(energies)
	#
	##	plt.scatter(180*alphas/np.pi,energies,label="T = "+str(round(T,2)),s=5)
	##	plt.legend()
	##
	##
	##plt.xlabel("$\\alpha$")
	##plt.ylabel("$F(\\alpha)$")
	##plt.title("$F(\\alpha)$ vs $\\alpha$ \n $\\chi = $"+str(Vs[0]/spe*(N-1)))
	##plt.savefig("Figures/Phase Transition/FreeEnergyVsAlpha.png",dpi=700)
	##plt.show()
	##
	#
	#
	#
	#################### Plots F(alpha)_min as a function of T ###################
	#for i in range(len(freeEnergies)):
	#	plt.scatter(freeEnergies[i][0],abs(180*alphas[np.max(np.where(freeEnergies[i][1]==np.min(freeEnergies[i][1]))[0])]/np.pi),c="b",s=10)
	##
	##
	##	
	#plt.xlabel("$T$")
	#plt.ylabel("$\\alpha_{min}$")
	#plt.title("$\\alpha_{min}$ vs T, $\\chi = $"+str(Vs[0]/spe*(N-1)))
	#plt.savefig("Figures/Phase Transition/AlphaMinVsT.png",dpi=700)
	#plt.show()
	#
	#
	#
	#################### Plot the heat capacity ###################
	#V = 0.05
	#spe = 0.5
	#N = 10
	#Ts = np.linspace(.1,1,80)
	#exactCapacities = []
	#for T in Ts:
	#	plt.scatter(T,HF_Heat_Capacity().HFHeatCapacity(spe,N,V,T,10**(-8)),c="b")
	#
	#	exactCapacities.append(ExactExpressions(spe,N,V,T).HeatCapacity())
	#plt.plot(Ts,exactCapacities,c='r')
	#plt.ylabel("$C_{V}$")
	#plt.xlabel("$T$")
	#plt.title("$C_{V}$ vs $T$ \n $\\chi = $"+str(V/spe*(N-1)))
	#plt.savefig("Figures/Phase Transition/HeatCapacity.png",dpi=700)
	#plt.show()
	
	
	#################### Plot T_c vs Chi ###################
#	Vs = np.linspace(0,1,100)
#	spe = 1
#	N = 10
#	Ns = np.arange(1,100,2)
#	Tcs = []
#	chis = Vs/spe*(N-1)
#	
#	for V in Vs:
#		Tcs.append(HF_Heat_Capacity().Tc(spe,N,V,10**(-8))[0])
#	
#	
#	plt.scatter(chis,Tcs,c='b')
#	plt.plot([1,1],[0,max(Tcs)],"r--",zorder=0)
#	plt.xlabel("$\\chi$")
#	plt.ylabel("$T_c$")
#	plt.title("$T_c$ vs $\\chi$")
#	plt.savefig("Figures/Phase Transition/Tc_vs_Chi.png",dpi=700)
#	plt.show()



