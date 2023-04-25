import numpy as np
from qiskit import QuantumCircuit
from qiskit import Aer, transpile
from qiskit.circuit import Parameter
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram



N = 4              #number of qubits
T = 12*np.pi         #final time
L = 200             #number of time steps
iter = 8            #number of iterations

ts = np.linspace(0, T, L)

def scaling(x):
    return x #quadratic scaling

def a(t):
    a = scaling(t)
    return a

def b(t):
    b = 1 - scaling(t)
    return b


qc = QuantumCircuit(N, N)
for qubit in range(0, N): # creating uniform superposition
    qc.h(qubit)

J = np.zeros(N-1)
for i in range(N-1):
    J[i] = np.random.choice([1, -1])
print(J)
dt = Parameter('dt')

for k in range(iter):
    for i in range(0, N - 1):
        # ZZ
        # qc.cnot(i, i+1)
        # qc.rz(2 * dt*J[i], i+1)
        # qc.cnot(i, i+1)

        qc.rzz(2 * np.pi * a(k/iter) * J[i] * dt, i, i+1)

    for i in range(0, N):      # H_D = uniform X field
        qc.rx(2 * np.pi * b(k/iter) * dt, i)

    qc.measure_all()
print(qc)
q_circs = [qc.assign_parameters({dt: a}, inplace=False) for a in ts]
num_shots_per_point = 1024

sim = Aer.get_backend('aer_simulator')
t_qc = transpile(q_circs, sim)
#
counts = sim.run(t_qc, shots=num_shots_per_point).result().get_counts()
# # print(counts)
# plot_histogram(counts)

expectation_z = np.zeros(L)
for i in range(L):
    count = counts[i]
    for key in count:
        bitstring = key
        for m in range(N-1):
            expectation_z[i] += J[m]*(int(count[key]) * (-1) ** (int(bitstring[m]) + int(bitstring[m+1])))
    expectation_z[i] = expectation_z[i] / num_shots_per_point

H = (expectation_z)


plt.plot(ts, H)
plt.title("Disordered Ising Model")
plt.xlabel('Time')
plt.ylabel('Expectation value of H')
plt.show()

