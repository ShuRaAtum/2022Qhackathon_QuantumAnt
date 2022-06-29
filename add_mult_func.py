import numpy as np

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.providers.aer import AerSimulator
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *
from qiskit.providers.aer import QasmSimulator

import time

from qiskit import *
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *
from qiskit.circuit.library import GroverOperator, ZGate
from qiskit.circuit import *
from qiskit.providers.aer.noise import NoiseModel

import numpy as np

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *
from qiskit.providers.aer import QasmSimulator
from qiskit.tools import job_monitor

import time


from qiskit import *
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *
from qiskit.circuit.library import GroverOperator, ZGate
from qiskit.circuit import *

from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
from qiskit.test.mock import FakeWashington


# Loading your IBM Quantum acc

# Loading your IBM Quantum account(s)

# True : cloud / False : simul
CLOUD_SIMUL = False

def MAJ(circuit, a, b, c):
    circuit.cx(a, b)
    circuit.cx(a, c)
    circuit.ccx(c, b, a)

def UMA(circuit, a, b, c):
    circuit.ccx(c, b, a)
    circuit.cx(a, c)
    circuit.cx(c, b)

def MAJ_reverse(circuit, a, b, c):
    circuit.ccx(c, b, a)
    circuit.cx(a, c)
    circuit.cx(a, b)

def UMA_reverse(circuit, a, b, c):
    circuit.cx(c, b)
    circuit.cx(a, c)
    circuit.ccx(c, b, a)

def ADD(circuit, a, b, c0, carry):  # Quantum ripple carry adder
    MAJ(circuit, a[0], b[0], c0)
    for i in range(3):
        MAJ(circuit, a[(i + + 1)], b[(i + + 1)], a[(+i)])
    circuit.cx(a[3], carry)
    for i in range(3):
        UMA(circuit, a[(3 - i)], b[(3 - i)], a[(3 - 1 - i)])
    UMA(circuit, a[0], b[0], c0)

def ADD_reverse(circuit, a, b, c0):
    UMA_reverse(circuit, a[0], b[0], c0)
    for i in range(3):
        UMA_reverse(circuit, a[(i + + 1)], b[(i + + 1)], a[(+i)])
    for i in range(3):
        MAJ_reverse(circuit, a[(3 - i)], b[(3 - i)], a[(3 - 1 - i)])
    MAJ_reverse(circuit, a[0], b[0], c0)

def ADD2(circuit, a, b, c0, carry):  # Quantum ripple carry adder
    MAJ(circuit, a[0], b[0], c0)
    for i in range(1):
        MAJ(circuit, a[(i + + 1)], b[(i + + 1)], a[(+i)])
    circuit.cx(a[1], carry)
    for i in range(1):
        UMA(circuit, a[(1 - i)], b[(1 - i)], a[(1 - 1 - i)])
    UMA(circuit, a[0], b[0], c0)


#안쓰는 함수
def addition2(operand_a, operand_b):
    a = QuantumRegister(2)
    b = QuantumRegister(2)
    carry = QuantumRegister(1)
    c0 = QuantumRegister(1)
    res = ClassicalRegister(3)

    circuit1 = QuantumCircuit(a, b, carry, c0, res)

    Round_constant_XOR(circuit1, a, operand_a, 2)
    Round_constant_XOR(circuit1, b, operand_b, 2)

    # Grover search
    ADD2(circuit1, a, b, c0, carry)

    circuit1.measure(b[0], res[0])
    circuit1.measure(b[1], res[1])
    circuit1.measure(carry, res[2])

    start = time.time()
    '''
    if CLOUD_SIMUL:
        backend = provider.get_backend('ibmq_manhattan')
    else:
        backend = BasicAer.get_backend('qasm_simulator')    # the device to run on
    result = execute(circuit1, backend, shots=5).result()
    counts = result.get_counts(circuit1)
    '''
    statevector_simulator = AerSimulator(method='statevector')

    # Transpile circuit for backend
    tcirc = transpile(circuit1, statevector_simulator)

    # Try and run circuit
    result = statevector_simulator.run(tcirc, shots=100).result()

    # result = execute(circuit1, backend, shots=1000).result()
    counts = result.get_counts(circuit1)

    # print("time : ", time.time() - start)
    circuit1 = circuit1.decompose()
    print("Decomposed depth : ", circuit1.depth())
    print(counts)


def addition4(operand_a, operand_b):
    a = QuantumRegister(4)
    b = QuantumRegister(4)
    carry = QuantumRegister(1)
    c0 = QuantumRegister(1)
    res = ClassicalRegister(5)

    circuit1 = QuantumCircuit(a, b, carry, c0, res)

    Round_constant_XOR(circuit1, a, operand_a, 4)
    Round_constant_XOR(circuit1, b, operand_b, 4)

    # Grover search
    ADD(circuit1, a, b, c0, carry)

    circuit1.measure([b[0], b[1], b[2], b[3]], [res[0], res[1], res[2], res[3]])
    circuit1.measure(carry, res[4])

    start = time.time()
    '''
    if CLOUD_SIMUL:
        backend = provider.get_backend('ibmq_manhattan')
    else:
        backend = BasicAer.get_backend('qasm_simulator')    # the device to run on
    result = execute(circuit1, backend, shots=1000).result()
    counts = result.get_counts(circuit1)
    
    statevector_simulator = AerSimulator(method='statevector')

    # Transpile circuit for backend
    tcirc = transpile(circuit1, statevector_simulator)

    # Try and run circuit
    result = statevector_simulator.run(tcirc, shots=5).result()

    # result = execute(circuit1, backend, shots=1000).result()
    counts = result.get_counts(circuit1)
    '''

    if CLOUD_SIMUL:
        backend = FakeWashington()
        # provider.backends()
        result = execute(circuit1, backend=backend).result()
    else:
        statevector_simulator = AerSimulator(method='statevector')
        # Transpile circuit for backend
        tcirc = transpile(circuit1, statevector_simulator)
        # Try and run circuit
        result = statevector_simulator.run(tcirc, shots=100).result()

    counts = result.get_counts(circuit1)

    # print("time : ", time.time() - start)
    circuit1 = circuit1.decompose()
    print("Decomposed depth : ", circuit1.depth())
    print(counts)
    # plot_histogram(counts).show()

    ret = int("0b" + str(counts.most_frequent()), 2)
    depth_return = circuit1.depth()
    return counts, depth_return, ret

#안쓰는 함수
def Karatsuba_Toffoli_Depth_1_3bit(operand_a, operand_b):
    n = 3
    a = QuantumRegister(n)
    b = QuantumRegister(n)
    c = QuantumRegister(6)
    test = QuantumRegister(8)
    ar0 = QuantumRegister(1)
    br0 = QuantumRegister(1)

    a0a1 = QuantumRegister(1)
    b0b1 = QuantumRegister(1)

    ar0ar1 = QuantumRegister(1)
    br0br1 = QuantumRegister(1)

    res = ClassicalRegister(3)

    circuit1 = QuantumCircuit(a, b, c, ar0, br0, a0a1, b0b1, ar0ar1, br0br1, res)
    # noise_model = NoiseModel.from_backend(backend)

    Round_constant_XOR(circuit1, a, operand_a, n)
    Round_constant_XOR(circuit1, b, operand_b, n)
    # Multiplication operands setting
    # low-part

    store_operand(circuit1, a[0], a[1], a0a1)
    store_operand(circuit1, b[0], b[1], b0b1)

    # middle-part
    store_operand(circuit1, a[0], a[2], ar0)
    # store_operand(circuit1, a[1], a[3], ar1)
    store_operand(circuit1, b[0], b[2], br0)
    # store_operand(circuit1, b[1], b[3], br1)

    # middle-inner-part
    store_operand(circuit1, ar0, a[1], ar0ar1)
    store_operand(circuit1, br0, b[1], br0br1)

    # high-part
    # store_operand(circuit1, a[2], a[3], a2a3)
    # store_operand(circuit1, b[2], b[3], b2b3)

    # Multiplication
    # low-part
    circuit1.ccx(a[0], b[0], c[0])
    circuit1.ccx(a0a1, b0b1, c[1])
    circuit1.ccx(a[1], b[1], c[2])

    # middle-part
    circuit1.ccx(ar0, br0, c[3])
    circuit1.ccx(ar0ar1, br0br1, c[4])
    # circuit1.ccx(ar1, br1, c[5])

    # high-part
    circuit1.ccx(a[2], b[2], c[5])
    # circuit1.ccx(a2a3, b2b3, c[7])
    # circuit1.ccx(a[3], b[3], c[8])

    # Combine (depth 5)
    # 1
    circuit1.cx(c[0], c[1])
    circuit1.cx(c[1], c[4])
    circuit1.cx(c[2], c[1])

    # 2
    circuit1.cx(c[3], c[4])
    circuit1.cx(c[3], c[2])
    circuit1.cx(c[0], c[2])
    circuit1.cx(c[5], c[2])

    output = []
    output.append(c[0])
    output.append(c[1])
    output.append(c[2])
    output.append(c[4])
    output.append(c[5])

    # Modular(circuit1, output)
    circuit1.cx(output[3], output[0])
    circuit1.cx(output[3], output[1])

    circuit1.cx(output[4], output[0])
    circuit1.cx(output[4], output[1])

    # print
    circuit1.measure(output[0], res[0])
    circuit1.measure(output[1], res[1])
    circuit1.measure(output[2], res[2])

    start = time.time()
    '''
    if CLOUD_SIMUL:
        backend = provider.get_backend('ibmq_manhattan')
    else:
        backend = BasicAer.get_backend('qasm_simulator')    # the device to run on
    '''

    statevector_simulator = AerSimulator(method='statevector')

    # Transpile circuit for backend
    tcirc = transpile(circuit1, statevector_simulator)

    # Try and run circuit
    result = statevector_simulator.run(tcirc, shots=5).result()

    # result = execute(circuit1, backend, shots=1000).result()
    counts = result.get_counts(circuit1)

    circuit1 = circuit1.decompose()
    print("Decomposed depth : ", circuit1.depth())
    print(counts)


def Karatsuba_Toffoli_Depth_1_4bit(operand_a, operand_b):
    n = 4
    a = QuantumRegister(n)
    b = QuantumRegister(n)
    c = QuantumRegister(9)

    ar0 = QuantumRegister(1)
    br0 = QuantumRegister(1)

    ar1 = QuantumRegister(1)
    br1 = QuantumRegister(1)

    a0a1 = QuantumRegister(1)
    b0b1 = QuantumRegister(1)

    ar0ar1 = QuantumRegister(1)
    br0br1 = QuantumRegister(1)

    a2a3 = QuantumRegister(1)
    b2b3 = QuantumRegister(1)

    res = ClassicalRegister(4)

    circuit1 = QuantumCircuit(a, b, c, ar0, br0, ar1, br1, a2a3, b2b3, a0a1, b0b1, ar0ar1, br0br1, res)

    Round_constant_XOR(circuit1, a, operand_a, n)
    Round_constant_XOR(circuit1, b, operand_b, n)

    # Multiplication operands setting
    # low-part
    store_operand(circuit1, a[0], a[1], a0a1)
    store_operand(circuit1, b[0], b[1], b0b1)

    # middle-part
    store_operand(circuit1, a[0], a[2], ar0)
    store_operand(circuit1, a[1], a[3], ar1)
    store_operand(circuit1, b[0], b[2], br0)
    store_operand(circuit1, b[1], b[3], br1)

    # middle-inner-part
    store_operand(circuit1, ar0, ar1, ar0ar1)
    store_operand(circuit1, br0, br1, br0br1)

    # high-part
    store_operand(circuit1, a[2], a[3], a2a3)
    store_operand(circuit1, b[2], b[3], b2b3)

    # Multiplication
    # low-part
    circuit1.ccx(a[0], b[0], c[0])
    circuit1.ccx(a0a1, b0b1, c[1])
    circuit1.ccx(a[1], b[1], c[2])

    # middle-part
    circuit1.ccx(ar0, br0, c[3])
    circuit1.ccx(ar0ar1, br0br1, c[4])
    circuit1.ccx(ar1, br1, c[5])

    # high-part
    circuit1.ccx(a[2], b[2], c[6])
    circuit1.ccx(a2a3, b2b3, c[7])
    circuit1.ccx(a[3], b[3], c[8])

    # Combine (depth 5)
    # 1

    circuit1.cx(c[0], c[1])
    circuit1.cx(c[2], c[1])

    circuit1.cx(c[6], c[7])
    circuit1.cx(c[8], c[7])

    # 2
    circuit1.cx(c[3], c[4])
    circuit1.cx(c[5], c[4])

    circuit1.cx(c[0], c[3])
    circuit1.cx(c[1], c[4])

    circuit1.cx(c[2], c[5])
    circuit1.cx(c[6], c[3])

    circuit1.cx(c[7], c[4])
    circuit1.cx(c[8], c[5])

    circuit1.cx(c[2], c[3])
    circuit1.cx(c[6], c[5])

    output = []
    output.append(c[0])
    output.append(c[1])
    output.append(c[3])
    output.append(c[4])
    output.append(c[5])
    output.append(c[7])
    output.append(c[8])

    Modular(circuit1, output)

    # print
    circuit1.measure(output[0], res[0])
    circuit1.measure(output[1], res[1])
    circuit1.measure(output[2], res[2])
    circuit1.measure(output[3], res[3])

    if CLOUD_SIMUL:
        backend = FakeWashington()
        # provider.backends()
        result = execute(circuit1, backend=backend).result()
    else:
        statevector_simulator = AerSimulator(method='statevector')
        # Transpile circuit for backend
        tcirc = transpile(circuit1, statevector_simulator)
        # Try and run circuit
        result = statevector_simulator.run(tcirc, shots=100).result()

    counts = result.get_counts(circuit1)

    circuit1 = circuit1.decompose()
    print("Decomposed depth : ", circuit1.depth())
    print(counts)
    ret = int("0b" + str(counts.most_frequent()), 2)
    depth_return = circuit1.depth()
    return counts, depth_return, ret

def Schoolbook_Mul(operand_a, operand_b):
    n = 4
    a = QuantumRegister(n)
    b = QuantumRegister(n)
    c = QuantumRegister(n)

    res = ClassicalRegister(4)

    circuit1 = QuantumCircuit(a, b, c, res)

    Round_constant_XOR(circuit1, a, operand_a, n)
    Round_constant_XOR(circuit1, b, operand_b, n)

    # Hight part
    circuit1.ccx(a[3], b[3], c[3])

    circuit1.ccx(a[3], b[2], c[2])
    circuit1.ccx(a[2], b[3], c[2])

    circuit1.ccx(a[3], b[1], c[1])
    circuit1.ccx(a[1], b[3], c[1])
    circuit1.ccx(a[2], b[2], c[1])

    circuit1.cx(c[1], c[0])
    circuit1.cx(c[2], c[1])
    circuit1.cx(c[3], c[2])

    # Low part
    circuit1.ccx(a[0], b[0], c[0])

    circuit1.ccx(a[1], b[0], c[1])
    circuit1.ccx(a[0], b[1], c[1])

    circuit1.ccx(a[2], b[0], c[2])
    circuit1.ccx(a[1], b[1], c[2])
    circuit1.ccx(a[0], b[2], c[2])

    circuit1.ccx(a[0], b[3], c[3])
    circuit1.ccx(a[3], b[0], c[3])
    circuit1.ccx(a[2], b[1], c[3])
    circuit1.ccx(a[1], b[2], c[3])

    # print
    circuit1.measure(c[0], res[0])
    circuit1.measure(c[1], res[1])
    circuit1.measure(c[2], res[2])
    circuit1.measure(c[3], res[3])

    if CLOUD_SIMUL:
        backend = FakeWashington()
        # provider.backends()
        result = execute(circuit1, backend=backend).result()
    else:
        statevector_simulator = AerSimulator(method='statevector')
        # Transpile circuit for backend
        tcirc = transpile(circuit1, statevector_simulator)
        # Try and run circuit
        result = statevector_simulator.run(tcirc, shots=100).result()

    counts = result.get_counts(circuit1)

    circuit1 = circuit1.decompose()
    print("Decomposed depth : ", circuit1.depth())
    print(counts)
    ret = int("0b" + str(counts.most_frequent()), 2)
    depth_return = circuit1.depth()
    return counts, depth_return, ret


def Karatsuba_4bit(operand_a, operand_b):
    n = 4
    a = QuantumRegister(n)
    b = QuantumRegister(n)
    c = QuantumRegister(9)

    res = ClassicalRegister(4)

    circuit1 = QuantumCircuit(a, b, c, res)

    Round_constant_XOR(circuit1, a, operand_a, n)
    Round_constant_XOR(circuit1, b, operand_b, n)

    circuit1.ccx(a[0], b[0], c[0])
    circuit1.ccx(a[1], b[1], c[2])
    circuit1.ccx(a[2], b[2], c[6])
    circuit1.ccx(a[3], b[3], c[8])

    circuit1.cx(a[1], a[0])
    circuit1.cx(b[1], b[0])
    circuit1.ccx(a[0], b[0], c[1])

    circuit1.cx(a[3], a[2])
    circuit1.cx(b[3], b[2])
    circuit1.ccx(a[2], b[2], c[7])

    circuit1.cx(a[2], a[0])
    circuit1.cx(b[2], b[0])
    circuit1.ccx(a[0], b[0], c[4])

    circuit1.cx(a[3], a[1])
    circuit1.cx(b[3], b[1])
    circuit1.ccx(a[1], b[1], c[5])

    circuit1.cx(a[1], a[0])
    circuit1.cx(b[1], b[0])
    circuit1.ccx(a[0], b[0], c[3])

    # Modular
    circuit1.cx(c[0], c[2])
    circuit1.cx(c[2], c[3])
    circuit1.cx(c[7], c[1])
    circuit1.cx(c[5], c[1])
    circuit1.cx(c[6], c[8])
    circuit1.cx(c[5], c[8])

    circuit1.cx(c[2], c[8])
    circuit1.cx(c[4], c[6])
    circuit1.cx(c[3], c[7])
    circuit1.cx(c[3], c[6])
    circuit1.cx(c[1], c[0])
    circuit1.cx(c[1], c[6])

    circuit1.measure(c[8], res[0])
    circuit1.measure(c[0], res[1])
    circuit1.measure(c[7], res[2])
    circuit1.measure(c[6], res[3])

    if CLOUD_SIMUL:
        backend = FakeWashington()
        # provider.backends()
        result = execute(circuit1, backend=backend).result()
    else:
        statevector_simulator = AerSimulator(method='statevector')
        # Transpile circuit for backend
        tcirc = transpile(circuit1, statevector_simulator)
        # Try and run circuit
        result = statevector_simulator.run(tcirc, shots=100).result()

    counts = result.get_counts(circuit1)

    circuit1 = circuit1.decompose()
    print("Decomposed depth : ", circuit1.depth())
    print(counts)
    ret = int("0b" + str(counts.most_frequent()), 2)
    depth_return = circuit1.depth()
    return counts, depth_return, ret


def CDKM(operand_a, operand_b):
    n = 4
    a = QuantumRegister(4)  # a
    b = QuantumRegister(4)  # b
    c = QuantumRegister(1)  # carry qubit
    z = QuantumRegister(1)
    res = ClassicalRegister(5)

    circuit1 = QuantumCircuit(a, b, c, z, res)

    Round_constant_XOR(circuit1, a, operand_a, n)
    Round_constant_XOR(circuit1, b, operand_b, n)

    for i in range(1, n):
        circuit1.cx(a[i], b[i])

    circuit1.cx(a[1], c)
    circuit1.ccx(a[0], b[0], c)
    circuit1.cx(a[2], a[1])
    circuit1.ccx(c, b[1], a[1])
    circuit1.cx(a[3], a[2])

    for i in range(2, n - 2):
        circuit1.ccx(a[i - 1], b[i], a[i])
        circuit1.cx(a[i + 2], a[i + 1])

    circuit1.ccx(a[n - 3], b[n - 2], a[n - 2])
    circuit1.cx(a[n - 1], z)
    circuit1.ccx(a[n - 2], b[n - 1], z)

    for i in range(1, n - 1):
        circuit1.x(b[i])

    circuit1.cx(c, b[1])

    for i in range(2, n):
        circuit1.cx(a[i - 1], b[i])

    circuit1.ccx(a[n - 3], b[n - 2], a[n - 2])

    for i in range(n - 3, 1, -1):
        circuit1.ccx(a[i - 1], b[i], a[i])
        circuit1.cx(a[i + 2], a[i + 1])
        circuit1.x(b[i + 1])

    circuit1.ccx(c, b[1], a[1])
    circuit1.cx(a[3], a[2])
    circuit1.x(b[2])

    circuit1.ccx(a[0], b[0], c)
    circuit1.cx(a[2], a[1])
    circuit1.x(b[1])

    circuit1.cx(a[1], c)

    for i in range(0, n):
        circuit1.cx(a[i], b[i])

    circuit1.measure(b[0], res[0])
    circuit1.measure(b[1], res[1])
    circuit1.measure(b[2], res[2])
    circuit1.measure(b[3], res[3])
    circuit1.measure(z, res[4])

    if CLOUD_SIMUL:
        backend = FakeWashington()
        # provider.backends()
        result = execute(circuit1, backend=backend).result()
    else:
        statevector_simulator = AerSimulator(method='statevector')
        # Transpile circuit for backend
        tcirc = transpile(circuit1, statevector_simulator)
        # Try and run circuit
        result = statevector_simulator.run(tcirc, shots=100).result()

    counts = result.get_counts(circuit1)

    circuit1 = circuit1.decompose()
    print("Decomposed depth : ", circuit1.depth())
    print(counts)
    ret = int("0b" + str(counts.most_frequent()), 2)
    depth_return = circuit1.depth()
    return counts, depth_return, ret


def Inversion(operand_a):
    n = 4
    a = QuantumRegister(4)  # a
    a0 = QuantumRegister(4)
    c0 = QuantumRegister(4)
    c1 = QuantumRegister(4)

    res = ClassicalRegister(4)
    circuit1 = QuantumCircuit(a, a0, c0, c1, res)

    Round_constant_XOR(circuit1, a, operand_a, n)
    Round_constant_XOR(circuit1, a0, operand_a, n)

    a = Squaring_0(circuit1, a)  # x^2

    # x^2 * x
    circuit1.ccx(a[3], a0[3], c0[3])

    circuit1.ccx(a[3], a0[2], c0[2])
    circuit1.ccx(a[2], a0[3], c0[2])

    circuit1.ccx(a[3], a0[1], c0[1])
    circuit1.ccx(a[1], a0[3], c0[1])
    circuit1.ccx(a[2], a0[2], c0[1])

    circuit1.cx(c0[1], c0[0])
    circuit1.cx(c0[2], c0[1])
    circuit1.cx(c0[3], c0[2])

    # Low part
    circuit1.ccx(a[0], a0[0], c0[0])

    circuit1.ccx(a[1], a0[0], c0[1])
    circuit1.ccx(a[0], a0[1], c0[1])

    circuit1.ccx(a[2], a0[0], c0[2])
    circuit1.ccx(a[1], a0[1], c0[2])
    circuit1.ccx(a[0], a0[2], c0[2])

    circuit1.ccx(a[0], a0[3], c0[3])
    circuit1.ccx(a[3], a0[0], c0[3])
    circuit1.ccx(a[2], a0[1], c0[3])
    circuit1.ccx(a[1], a0[2], c0[3])

    c0 = Squaring_0(circuit1, c0)  # output = x^6

    a = Squaring_0(circuit1, a)  # x^4
    a = Squaring_0(circuit1, a)  # x^8

    circuit1.ccx(a[3], c0[3], c1[3])

    circuit1.ccx(a[3], c0[2], c1[2])
    circuit1.ccx(a[2], c0[3], c1[2])

    circuit1.ccx(a[3], c0[1], c1[1])
    circuit1.ccx(a[1], c0[3], c1[1])
    circuit1.ccx(a[2], c0[2], c1[1])

    circuit1.cx(c1[1], c1[0])
    circuit1.cx(c1[2], c1[1])
    circuit1.cx(c1[3], c1[2])

    # Low part
    circuit1.ccx(a[0], c0[0], c1[0])

    circuit1.ccx(a[1], c0[0], c1[1])
    circuit1.ccx(a[0], c0[1], c1[1])

    circuit1.ccx(a[2], c0[0], c1[2])
    circuit1.ccx(a[1], c0[1], c1[2])
    circuit1.ccx(a[0], c0[2], c1[2])

    circuit1.ccx(a[0], c0[3], c1[3])
    circuit1.ccx(a[3], c0[0], c1[3])
    circuit1.ccx(a[2], c0[1], c1[3])
    circuit1.ccx(a[1], c0[2], c1[3])

    circuit1.measure(c1[0], res[0])
    circuit1.measure(c1[1], res[1])
    circuit1.measure(c1[2], res[2])
    circuit1.measure(c1[3], res[3])

    if CLOUD_SIMUL:
        backend = FakeWashington()
        # provider.backends()
        result = execute(circuit1, backend=backend).result()
    else:
        statevector_simulator = AerSimulator(method='statevector')
        # Transpile circuit for backend
        tcirc = transpile(circuit1, statevector_simulator)
        # Try and run circuit
        result = statevector_simulator.run(tcirc, shots=100).result()

    counts = result.get_counts(circuit1)

    circuit1 = circuit1.decompose()
    print("Decomposed depth : ", circuit1.depth())
    print(counts)
    ret = int("0b" + str(counts.most_frequent()), 2)
    depth_return = circuit1.depth()
    return counts, depth_return, ret


def Squaring_0(circuit, a):
    circuit.cx(a[3], a[1])
    circuit.cx(a[2], a[0])  # 0123->0213
    output = []
    output.append(a[0])
    output.append(a[2])
    output.append(a[1])
    output.append(a[3])

    return output


def Squaring(operand_a):
    n = 4
    a = QuantumRegister(4)  # a
    res = ClassicalRegister(4)

    circuit1 = QuantumCircuit(a, res)

    Round_constant_XOR(circuit1, a, operand_a, n)

    circuit1.cx(a[3], a[1])
    circuit1.cx(a[2], a[0])

    output = []
    output.append(a[0])
    output.append(a[2])
    output.append(a[1])
    output.append(a[3])

    circuit1.measure(output[0], res[0])
    circuit1.measure(output[1], res[1])
    circuit1.measure(output[2], res[2])
    circuit1.measure(output[3], res[3])

    if CLOUD_SIMUL:
        backend = FakeWashington()
        # provider.backends()
        result = execute(circuit1, backend=backend).result()
    else:
        statevector_simulator = AerSimulator(method='statevector')
        # Transpile circuit for backend
        tcirc = transpile(circuit1, statevector_simulator)
        # Try and run circuit
        result = statevector_simulator.run(tcirc, shots=100).result()

    counts = result.get_counts(circuit1)

    circuit1 = circuit1.decompose()
    print("Decomposed depth : ", circuit1.depth())
    print(counts)
    ret = int("0b" + str(counts.most_frequent()), 2)
    depth_return = circuit1.depth()
    return counts, depth_return, ret


def addition4_correction(operand_a, operand_b):
    a = QuantumRegister(4)
    b = QuantumRegister(4)
    carry = QuantumRegister(1)
    c0 = QuantumRegister(1)
    res = ClassicalRegister(5)

    cq_0 = QuantumRegister(2)
    lq_0 = QuantumRegister(2)
    cq_1 = QuantumRegister(2)
    lq_1 = QuantumRegister(2)
    cq_2 = QuantumRegister(2)
    lq_2 = QuantumRegister(2)

    circuit1 = QuantumCircuit(a, b, carry, c0, res, cq_0, cq_1, cq_2, lq_0, lq_1, lq_2)

    circuit1.cx(a[0], cq_0[0])
    circuit1.cx(a[0], cq_0[1])
    circuit1.cx(a[1], cq_1[0])
    circuit1.cx(a[1], cq_1[1])

    circuit1.cx(b[1], cq_2[0])
    circuit1.cx(b[1], cq_2[1])

    Round_constant_XOR(circuit1, a, operand_a, 4)
    Round_constant_XOR(circuit1, b, operand_b, 4)

    circuit1.x(cq_0[0])
    circuit1.x(cq_0[1])

    circuit1.x(cq_1[0])
    circuit1.x(cq_1[1])

    circuit1.x(cq_2[0])
    circuit1.x(cq_2[1])

    correct(circuit1, a[0], cq_0, lq_0)
    correct(circuit1, a[1], cq_1, lq_1)
    correct(circuit1, b[1], cq_2, lq_2)

    # Omit Decoding

    # Perform Addition
    ADD4(circuit1, a, b, c0, carry)

    circuit1.measure([b[0], b[1], b[2], b[3]], [res[0], res[1], res[2], res[3]])
    circuit1.measure(carry, res[4])

    if CLOUD_SIMUL:
        backend = FakeWashington()
        # provider.backends()
        result = execute(circuit1, backend=backend).result()
    else:
        statevector_simulator = AerSimulator(method='statevector')
        # Transpile circuit for backend
        tcirc = transpile(circuit1, statevector_simulator)
        # Try and run circuit
        result = statevector_simulator.run(tcirc, shots=100).result()

    counts = result.get_counts(circuit1)

    circuit1 = circuit1.decompose()
    print("Decomposed depth : ", circuit1.depth())
    print(counts)
    ret = int("0b" + str(counts.most_frequent()), 2)
    depth_return = circuit1.depth()
    return counts, depth_return, ret


def encode(circuit, qubit, cq):
    circuit.cx(qubit, cq[0])
    circuit.cx(qubit, cq[1])


def correct(circuit, qubit, cq, lq):
    circuit.cx(qubit, lq[0])
    circuit.cx(qubit, lq[1])
    circuit.cx(cq[0], lq[0])
    circuit.cx(cq[1], lq[1])
    circuit.ccx(lq[0], lq[1], qubit)
    circuit.ccx(lq[0], lq[1], cq[0])
    circuit.ccx(lq[0], lq[1], cq[1])


def decode(circuit, qubit, cq):
    circuit.cx(qubit, cq[0])
    circuit.cx(qubit, cq[1])

def ADD4(circuit, a, b, c0, carry):  # Quantum ripple carry adder
    MAJ(circuit, a[0], b[0], c0)
    for i in range(3):
        MAJ(circuit, a[(i + + 1)], b[(i + + 1)], a[(+i)])
    circuit.cx(a[3], carry)
    for i in range(3):
        UMA(circuit, a[(3 - i)], b[(3 - i)], a[(3 - 1 - i)])
    UMA(circuit, a[0], b[0], c0)


def Modular(circuit, c):
    circuit.cx(c[4], c[0])
    circuit.cx(c[4], c[1])

    circuit.cx(c[5], c[1])
    circuit.cx(c[5], c[2])

    circuit.cx(c[6], c[2])
    circuit.cx(c[6], c[3])


def store_operand(circuit, a, b, c):
    circuit.cx(a, c)
    circuit.cx(b, c)


def Round_constant_XOR(circuit, k, rc, bit):
    for i in range(bit):
        if (rc >> i & 1):
            circuit.x(k[i])


def QCLA(operand_a, operand_b):
    n=4
    a = QuantumRegister(4)
    b = QuantumRegister(4)
    z = QuantumRegister(4)
    anc = QuantumRegister(1)
    res = ClassicalRegister(5)
    cir = QuantumCircuit(a, b, z, anc, res)

    Round_constant_XOR(cir, a, operand_a, n)
    Round_constant_XOR(cir, b, operand_b, n)

    for i in range(4):
        cir.ccx(a[i], b[i], z[i])

    for i in range(4):
        cir.cx(a[i], b[i])

    cir.ccx(b[2], b[3], anc)

    cir.ccx(z[0], b[1], z[1])
    cir.ccx(z[2], b[3], z[3])
    cir.ccx(z[1], anc, z[3])

    cir.ccx(z[1], b[2], z[2])

    cir.ccx(b[2], b[3], anc)

    for i in range(3):
        cir.cx(z[i], b[i + 1])

    for i in range(3):
        cir.x(b[i])

    for i in range(2):
        cir.cx(a[i + 1], b[i + 1])

    cir.ccx(z[1], b[2], z[2])

    cir.ccx(z[0], b[1], z[1])

    for i in range(2):
        cir.cx(a[i + 1], b[i + 1])

    for i in range(3):
        cir.ccx(a[i], b[i], z[i])

    for i in range(3):
        cir.x(b[i])

    cir.measure([b[0], b[1], b[2], b[3], z[3]], [res[0], res[1], res[2], res[3], res[4]])
    # circuit1.measure([p[0], p[1], p[2], p[3]],[res[0],res[1],res[2],res[3]])

    if CLOUD_SIMUL:
        backend = FakeWashington()
        # provider.backends()
        result = execute(cir, backend=backend).result()
    else:
        backend = BasicAer.get_backend('qasm_simulator')  # the device to run on
        result = execute(cir, backend, shots=5).result()

    counts = result.get_counts(cir)

    print(counts)

    ret = int("0b" + str(counts.most_frequent()), 2)
    depth_return = cir.depth()
    return counts, depth_return, ret
