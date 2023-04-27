import numpy as np
import matplotlib.pyplot as plt
from quimb import *
from scipy import sparse
from quimb.linalg.base_linalg import *
from qiskit import QuantumCircuit
from qiskit import Aer, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram
from scipy.linalg import expm
from qiskit.quantum_info import Operator
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)


N = 4  # Number of qubits

J = np.zeros((N, N))
for i in range(N - 1):
    J[i, i + 1] = np.random.choice([-1, 1])

# print(J)

I, X, Y, Z = (pauli(s) for s in 'IXYZ')


def dis_ising(J, N):
    M = 2 ** N  # DImension of hilbert space
    H = np.zeros((M, M))
    for i in range(N - 1):
        H = H + J[i, i + 1] * ikron(Z, dims=[2] * N, inds=[i, i + 1])
    return H


H = dis_ising(J, N)
# U = expm(H*dt)

gs = groundstate(H)
# print(gs)
# test_state = kron(up(), down(), up(), up())
# print(expec(test_state, H))

# m = np.zeros((N))
# for i in range(N):
#    rho_red = ptr(gs,dims = [2]*N, keep = i)
#    m[i] = expec(Z,rho_red)
# print(m)


analytical_groundstate_energy = expec(H, gs)
print("Lowest energy from exact Diagonalisation is = ", analytical_groundstate_energy)


def scaling(x):
    if x <= 0.5:
        y = 0.5 * np.sin(np.pi * x)
    elif x > 0.5:
        y = 1 - 0.5 * np.sin(np.pi * x)
    # y = (x-0.5)**3+1/2
    return y


# ts = np.linspace(0, 1, 100)
# scale = np.zeros(100)
# for i in range(100):
#     scale[i] = scaling(ts[i])
#
# plt.plot(ts, scale)
# plt.show()

def a(t):
    a = scaling(t)
    return a


def b(t):
    b = 1 - scaling(t)
    return b

# Create an empty noise model
noise_model = NoiseModel()

# Add depolarizing error to all single qubit u1, u2, u3 gates
error_pauli = depolarizing_error(0.0005, 1)
error_cnot = depolarizing_error(0.005, 1)
noise_model.add_all_qubit_quantum_error(error_pauli, ['rz', 'rx'])
noise_model.add_all_qubit_quantum_error(error_cnot, ['cnot'])


# when a random error is triggered, it's a Pauli X, Y or Z error with probability 1/3 each),
# and an average error rate of 0.5% per CNOT. The average error rate per single qubit gate
# should be 1/10th this, so 0.05% (probability 0.0005), again depolarizing noise.


# print(noise_model)

def GSA_approx(T, L, J):
    dt = T / L  # time step
    qc = QuantumCircuit(N)
    for qubit in range(0, N):  # creating uniform superposition
        qc.h(qubit)
    for k in range(L):
        for i in range(N - 2, -1, -1):
            j = i + 1
            qc.rzz(-1 * a(k * dt / T) * J[i, j] * dt, j, i)
        for i in range(0, N):  # H_D = uniform X field as the Driver Hamiltonian
            qc.rx(1 * b(k * dt / T) * dt, i)
    qc.measure_all()
    num_shots_per_point = 1024
    sim_noise = AerSimulator(noise_model=noise_model)
    # Transpile circuit for noisy basis gates
    circ_tnoise = transpile(qc, sim_noise)

    # Run and get counts
    result_bit_flip = sim_noise.run(circ_tnoise).result()
    counts = result_bit_flip.get_counts(0)

    # Plot noisy output
    plot_histogram(counts)
    expectation_z = 0
    for key in counts:
        bitstring = key
        for m in range(N - 1, 0, -1):
            expectation_z += J[N - m - 1, N - m] * (
                        int(counts[key]) * (-1) ** (int(bitstring[m - 1]) + int(bitstring[m])))
    expectation_z = expectation_z / num_shots_per_point
    return expectation_z

L = 8
T = 4 * np.pi
# GSA_approx(T,L,J)

print(GSA_approx(T, L, J))

# Llist = np.linspace(10,200,15,dtype=int)
# approx = np.zeros((15))
# for i in range(15):
#     print(i)
#     approx[i]=np.abs(np.abs(analytical_groundstate_energy)-np.abs(GSA_approx(T,Llist[i],J)))
#
# plt.plot(Llist,approx)
# plt.xlabel('L')
# plt.ylabel('ΔE')
# plt.title("Ground State approximation")
# plt.savefig("Groundstateapprox_vs_L(fixed T).pdf")
# plt.show()



