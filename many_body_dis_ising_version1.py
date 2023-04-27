import numpy as np
import matplotlib.pyplot as plt
from quimb import *
from scipy import sparse
from quimb.linalg.base_linalg import *
from qiskit import QuantumCircuit
from qiskit import Aer, transpile
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram
from scipy.linalg import expm
from qiskit.quantum_info import Operator


N = 12 #Number of qubits

J = np.zeros((N,N))
for i in range(N-1):
    J[i,i+1] = np.random.choice([-1, 1])

print(J)

I, X, Y, Z = (pauli(s) for s in 'IXYZ')

def dis_ising(J,N):
    M = 2**N # DImension of hilbert space
    H = np.zeros((M,M))
    for i in range(N-1):
        H = H + J[i,i+1]*ikron(Z,dims = [2]*N,inds = [i,i+1])
    return H

H = dis_ising(J, N)
# U = expm(H*dt)

gs = groundstate(H)
# print(gs)
# test_state = kron(up(), down(), up(), up())
# print(expec(test_state, H))

#m = np.zeros((N))
#for i in range(N):
#    rho_red = ptr(gs,dims = [2]*N, keep = i)
#    m[i] = expec(Z,rho_red)
#print(m)


analytical_groundstate_energy = expec(H,gs)
print("Lowest energy from exact Diagonalisation is = ", analytical_groundstate_energy)


def scaling(x):
        return x**2 #Linear Scaling
def a(t):
    a = scaling(t)
    return a
def b(t):
    b = 1 - scaling(t)
    return b

def GSA_approx(T,L,J,p):
    dt = T/L          # time step
    ts = np.linspace(0, T, L)
    qc = QuantumCircuit(N)
    for qubit in range(0, N): # creating uniform superposition
        qc.h(qubit)
    for k in range(L):
        for i in range(N-2, -1, -1):
            j = i+1
            qc.rzz(-np.pi * a(k/L) * J[i, j] * dt, j, i)
        for i in range(0, N):      # H_D = uniform X field as the Driver Hamiltonian
            qc.rx(np.pi * b(k/L) * dt, i)
    qc.measure_all()
    num_shots_per_point = 4000
    sim = Aer.get_backend('aer_simulator')
    t_qc = transpile(qc, sim)
    counts = sim.run(t_qc, shots=num_shots_per_point).result().get_counts()
    expectation_z = 0
    for key in counts:
        bitstring = key
        for m in range(N-1, 0, -1):
            expectation_z += J[N-m-1, N-m]*(int(counts[key]) * (-1) ** (int(bitstring[m-1]) + int(bitstring[m])))
    expectation_z = expectation_z / num_shots_per_point
    return expectation_z





T = 50*np.pi       #final time
L = 100
#number of time steps
GSA_approx(T,L,J)



T = 20*np.pi
Llist = np.linspace(10,200,15,dtype=int)
approx = np.zeros((15))
for i in range(15):
    print(i)
    approx[i]=np.abs(np.abs(analytical_groundstate_energy)-np.abs(GSA_approx(T,Llist[i],J)))
    
plt.plot(Llist,approx)
plt.xlabel('L')
plt.ylabel('ΔE')
plt.title("Ground State approximation")
plt.savefig("Groundstateapprox_vs_L(fixed T).pdf")
plt.show()

    

