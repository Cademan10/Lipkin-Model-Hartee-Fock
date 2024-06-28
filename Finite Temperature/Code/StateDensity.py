import numpy as np
import math
from numpy import log,exp,sqrt
import matplotlib.pyplot as plt 

params= {'text.latex.preamble' : r'\usepackages{amsmath,braket}'}
plt.rcParams.update(params)

#imports zero temperature code 
import sys
sys.path.append(r'/Users/cader/Desktop/Yale Research/Lipkin Model/Finite Temperature/Code')

from FiniteTempLipkinModel import FT_Lipkin_Model_HF,ExactExpressions, Lipkin_Model_HF, HF_Heat_Capacity


def Z_pp(spe,N,V,beta,NProj=False):
	return FT_Lipkin_Model_HF(spe,N,V,1/beta,10e-8).HartreeFock(partProj=True,projN=NProj)

chi = 1.8
spe = 0.5
#Ts = np.append(np.append(np.linspace(.05,0.15,100),np.linspace(.16,0.26,500)),np.linspace(0.27,3,100))
betas = np.linspace(.1,10,500)
Ts = 1/betas
N = int(input("N = "))
V = chi*spe/(N-1)

SP_Es = []
SP_Es_HF = []
SP_Es_HF_PP = []

SPrhos = []
SPrhos_HF = []
SPrhos_HF_PP = []

Cs = []
Cs_HF = []
Cs_HF_PP = []

Ss = []
Ss_HF = []
Ss_HF_PP = []

N_s = []

gsShift = True

lhf = Lipkin_Model_HF(spe,N,V)
Egs = lhf.Diagonalization()
Egs_HF = lhf.Numerical_Solver()

for i in range(len(betas)):
	beta = betas[i]
	T = 1/beta


	lexact = ExactExpressions(spe,N,V,T)

	E = lexact.ExactEnergy()
	C = lexact.HeatCapacity()
	PF = lexact.PartitonFunc()	

	Cs.append(C)
	Ss.append(E/T+log(PF))

	SPrhos.append(PF*exp(E*beta)/sqrt(2*np.pi*T**2*C))
	SP_Es.append(E)


	E,S = FT_Lipkin_Model_HF(spe,N,V,T,.00001).HartreeFock(entropy=True)
	C = HF_Heat_Capacity().HFHeatCapacity(spe,N,V,T,10**(-8))
	
	SPrhos_HF.append(exp(S)/sqrt(2*np.pi*T**2*C))
	SP_Es_HF.append(E)
	Ss_HF.append(S)
	Cs_HF.append(C)

	dEdBeta = 1
	i = 2

	#Chooses step-size such that heat capacity is positive 
	while dEdBeta>0:

		dBeta = 10**(-i)
		Z_pp_T,N_T = Z_pp(spe,N,V,beta,True)
		E_pp = -(Z_pp(spe,N,V,beta+dBeta)-Z_pp(spe,N,V,beta-dBeta))/(2*dBeta)
	
		E_pp_2 = -(Z_pp(spe,N,V,beta+2*dBeta)-Z_pp_T)/(2*dBeta)
		E_pp_1 = -(Z_pp_T-Z_pp(spe,N,V,beta-2*dBeta))/(2*dBeta)
		dEdBeta = (E_pp_2-E_pp_1)/(2*dBeta)

		i+=1

	
	N_s.append(N_T)
	Cs_HF_PP.append(-beta**2*dEdBeta)
	SP_Es_HF_PP.append(E_pp)
	S_HF_pp = Z_pp_T+E_pp*beta

	Ss_HF_PP.append(S_HF_pp)
	SPrhos_HF_PP.append(exp(S_HF_pp)/sqrt(-2*np.pi*dEdBeta))


### HF-SHELL derivative routine
def findiff(x,y):
	if len(x) != len(y):
		sys.stderr.write('Error: Derivative variables have unequal length')
		sys.exit(1)
	f = np.zeros(len(y))
	for i in range(0,len(y)-1):
		if i == 0:
			dx = x[1]-x[0]
			dy = y[1]-y[0]
			f[i] = dy/dx
		else:
			dx = x[i+1] - x[i-1]
			dy = y[i+1]-y[i-1]
			f[i] = dy/dx
	f[-1] = (y[-1]-y[-2])/(x[-1]-x[-2])
	return f

logZ_pp = []
for beta in betas:
	logZ_pp.append(Z_pp(spe,N,V,beta))

SP_Es_HF_SHELL =  -findiff(betas, logZ_pp)
Ss_HF_SHELL =  betas * SP_Es_HF_SHELL + logZ_pp
Cs_HF_SHELL =   -findiff(betas, SP_Es_HF_SHELL)/Ts**2
SPrhos_HF_SHELL = 1./(np.sqrt(2 * np.pi * Ts**2*np.abs(Cs_HF_SHELL))) * np.exp(Ss_HF_SHELL)


ks = np.arange(0+(N%2)/2,int(N/2)+1)

energies = []
degs = []
for k in ks:
	if k==N/2:
		degenergacy=1
	else:
			degenergacy =  math.factorial(N)**2/(math.factorial(int(N/2-k))**2*math.factorial(int(N/2+k))**2)\
-math.factorial(N)**2/(math.factorial(int(N/2-k-1))**2*math.factorial(int(N/2+k+1))**2)
	
	en = list(np.around(lexact.KSubspace(k),2))
	energies+=en
	degs+=list(degenergacy*np.ones(len(en)))

inds = np.array(energies).argsort()
energies = np.sort(energies)

degs = np.array(degs)[inds]
	
exact_rhos = []
final_energies = []


while len(energies)>0:
	E1 = energies[0]
	
	rho = 0

	final_energies.append(energies[0])
	while abs(E1-energies[0])<.001:
		energies = np.delete(energies,0)
		
		rho+=degs[0]

		degs = np.delete(degs,0)

		if len(energies)==0:
			break
		
	exact_rhos.append(rho)


indices = np.where(np.array(final_energies)<=.0001)


plt.plot((np.array(SP_Es)-int(gsShift)*Egs)/spe,SPrhos,"r",label="Exact (S.P. approxmation)")
plt.plot((np.array(SP_Es_HF)-int(gsShift)*Egs_HF)/spe,SPrhos_HF,"b--",label="HF")
plt.plot((np.array(SP_Es_HF_PP)-int(gsShift)*Egs_HF)/spe,SPrhos_HF_PP,color='green',ls="--",label="Particle-Projected HF")
plt.plot((np.array(SP_Es_HF_SHELL)-int(gsShift)*Egs_HF)/spe,SPrhos_HF_SHELL,color='purple',ls="--",label="Particle-Projected HF (HF-SHELL derivative)")
plt.hist((np.array(final_energies)[indices]-int(gsShift)*Egs)/spe,weights=np.array(exact_rhos)[indices],bins=int(len(np.array(exact_rhos)[indices])/4),histtype='step',lw=1,color='k')
plt.legend()
if gsShift==True:

	plt.xlabel("$(E-E_{0})/\\epsilon$")
	plt.title("Saddle-Point $\\rho(E)$ vs $E-E_{0}$\n $(V = $"+str(round(V,2))+", $\\epsilon = $"+str(spe)+", $N = $"+str(N)+"$)$")
else:
	plt.xlabel("$E/\\epsilon$")
	plt.title("$\\Saddle-Point rho(E)$ vs $E$\n $(V = $"+str(round(V,2))+", $\\epsilon = $"+str(spe)+", $N = $"+str(N)+"$)$")

plt.ylabel("$\\rho(E)$")

plt.yscale('log') 
plt.savefig("Figures/Level Density/StateDensity.png",dpi=700)
plt.show()

plt.plot(betas,(np.array(SP_Es)-int(gsShift)*Egs)/spe,"r",label="Exact")
plt.plot(betas,(np.array(SP_Es_HF)-int(gsShift)*Egs_HF)/spe,"b--",label="HF")
plt.plot(betas,(np.array(SP_Es_HF_PP)-int(gsShift)*Egs_HF)/spe,color='green',ls="--",label="Particle-Projected HF")
plt.plot(betas,(np.array(SP_Es_HF_SHELL)-int(gsShift)*Egs_HF)/spe,color='purple',ls="--",label="Particle-Projected HF (HF-SHELL derviative)")
plt.xlabel("$\\beta$")

if gsShift==True:
	plt.ylabel("$(E-E_{0})/\\epsilon$")
else:
	plt.ylabel("$E/\\epsilon$")
plt.title("$E$ vs $\\beta$ \n $(V = $"+str(round(V,2))+", $\\epsilon = $"+str(spe)+", $N = $"+str(N)+"$)$")
plt.legend()
plt.savefig("Figures/Level Density/EvsNuclearBeta.png",dpi=700)
plt.close()



plt.plot(betas,Cs,"r",label="Exact")
plt.plot(betas,Cs_HF,"b--",label="HF")
plt.plot(betas,Cs_HF_PP,color='green',ls="--",label="Particle-Projected HF")
plt.plot(betas,Cs_HF_SHELL,color='purple',ls="--",label="Particle-Projected HF (HF-SHELL derviative)")
plt.xlabel("$\\beta$")
plt.ylabel("$C_{v}$")
plt.title("$C_{v}$ vs $\\beta$ \n $(V = $"+str(round(V,2))+", $\\epsilon = $"+str(spe)+", $N = $"+str(N)+"$)$")
plt.legend()
plt.savefig("Figures/Level Density/NuclearHeatCapVsBeta.png",dpi=700)
plt.close()

#plt.scatter(betas,N_s,s=.1)
#plt.ylim(N-.000000001,N+.000000001)
#plt.ylabel("$\\langle{N}\\rangle_{Proj.}$")
#plt.xlabel("$\\beta$")
#ax = plt.gca()
#ax.get_yaxis().get_major_formatter().set_useOffset(False)
#plt.title("$\\langle{N}\\rangle_{Proj.}$ vs $\\beta$ \n $(N = $"+str(N)+"$)$")
#plt.tight_layout()
#plt.show()

plt.plot(betas,Ss,"r",label="Exact")
plt.plot(betas,Ss_HF,"b--",label="HF")
plt.plot(betas,Ss_HF_PP,color='green',ls="--",label="Particle-Projected HF")
plt.plot(betas,Ss_HF_SHELL,color='purple',ls="--",label="Particle-Projected HF (HF-SHELL derviative)")
plt.legend()
plt.xlabel("$\\beta$")
plt.ylabel("$S$")
plt.title("$S$ vs $\\beta$ \n $(V = $"+str(round(V,2))+", $\\epsilon = $"+str(spe)+", $N = $"+str(N)+"$)$")
plt.savefig("Figures/Level Density/NuclearEntropyVsBeta.png",dpi=700)
plt.close()







	



		



