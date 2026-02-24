import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)
from qiskit.visualization import plot_histogram

def build_U2bit_circuit():
    qc = QuantumCircuit(2)
    alpha = np.pi/4
    omega = 1.0
    tau = 1.0
    theta_rot = 2 * omega * tau
    nx = np.sin(alpha)
    nz = np.cos(alpha)
    beta = np.arctan2(nx, nz)
    
    # U_i (partícula)
    qc.ry(-beta, 1)
    qc.rz(theta_rot, 1)
    qc.ry(beta, 1)
    
    # S (scattering)
    theta_s = np.pi/3
    qc.x(0)
    qc.crz(theta_s, 0, 1)
    qc.x(0)
    qc.crx(theta_s, 0, 1)
    
    # U_i final (partícula)
    qc.ry(-beta, 1)
    qc.rz(theta_rot, 1)
    qc.ry(beta, 1)
    return qc.to_gate(label="U_2bit")

# ============================================================
# 1. CONSTRUCCIÓN DEL MODELO DE RUIDO (ibmqx4 aprox)
# ============================================================
def build_noise_model():
    """
    Construye un modelo de ruido térmico, depolarizador y de lectura
    para simular un entorno físico realista (como el chip IBM Tenerife de 2019).
    """
    noise_model = NoiseModel()
    
    # T1 y T2 en nanosegundos (típico de IBMQ 2019 ~ 50 microseconds)
    t1 = 50e3  
    t2 = 70e3  
    
    # Tiempos de duración g_time de las compuertas (ns)
    time_u1 = 0     # virtual
    time_u2 = 50    # microondas corta
    time_u3 = 100   # microondas larga
    time_cx = 300   # CNOT cross-resonance lenta
    
    # 1. Relajación térmica (Decoherencia con el entorno)
    err_u1 = thermal_relaxation_error(t1, t2, time_u1)
    err_u2 = thermal_relaxation_error(t1, t2, time_u2)
    err_u3 = thermal_relaxation_error(t1, t2, time_u3)
    err_cx = thermal_relaxation_error(t1, t2, time_cx).expand(
             thermal_relaxation_error(t1, t2, time_cx))
    
    # 2. Errores depolarizadores (Errores de calibración en las puertas)
    # Errores del paper: CNOT ~ 0.027, 1-Qubit ~ 0.001
    depol_1q = depolarizing_error(0.001, 1)
    depol_2q = depolarizing_error(0.027, 2)
    
    # Juntar térmico + depolarizador
    error_1q = err_u2.compose(depol_1q)
    error_2q = err_cx.compose(depol_2q)
    
    noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz', 'h', 'x', 'y', 'z', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'crx', 'crz'])
    
    # 3. Ruido de Lectura (Readout Error)
    # Según paper: r0~4.8%, r1~3.3%
    # La probabilidad de medir 0 dado 1 (u 1 dado 0) no es asimétrica pura usualmente en Qiskit
    # Usamos matrix [[P(0|0), P(1|0)], [P(0|1), P(1|1)]]
    probs = [[0.952, 0.048], [0.033, 0.967]]
    read_err = ReadoutError(probs)
    noise_model.add_all_qubit_readout_error(read_err)
    
    return noise_model

# ============================================================
# 2. EJECUCIÓN DEL CIRCUITO
# ============================================================
print("Construyendo experimento cuántico realista con Decoherencia Termal...")

qc = QuantumCircuit(2, 2)
U_gate = build_U2bit_circuit()

qc.append(U_gate, [0, 1])
qc.barrier()
qc.append(U_gate.inverse(), [0, 1]) # K operator
qc.barrier()
qc.append(U_gate, [0, 1])

qc.measure([0, 1], [0, 1])

# Compilación
simulator = AerSimulator()
# Base de IBM real para inyectar correctamente el ruido
compiled_qc = transpile(qc, simulator, basis_gates=['rx', 'ry', 'rz', 'cx'])

# 1. Simulación sin ruido (Ideal)
job_ideal = simulator.run(compiled_qc, shots=8192)
counts_ideal = job_ideal.result().get_counts()

# 2. Simulación con modelo de ruido físico de IBM Oruga (ibmqx4 aprox)
noise_model = build_noise_model()
simulator_noisy = AerSimulator(noise_model=noise_model)
job_noisy = simulator_noisy.run(compiled_qc, shots=8192)
counts_noisy = job_noisy.result().get_counts()

probs_ideal = {state: count/8192 for state, count in counts_ideal.items()}
probs_noisy = {state: count/8192 for state, count in counts_noisy.items()}

print(f"Probabilidad de éxito Ideal P(|00>): {probs_ideal.get('00', 0.0)*100:.1f}%")
print(f"Probabilidad de éxito Real (Ruido Físico) P(|00>): {probs_noisy.get('00', 0.0)*100:.1f}%")

# ============================================================
# 3. GRÁFICA COMPARATIVA
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Unificar estados dicts 
all_states = ['00', '01', '10', '11']
counts_ideal_full = {s: counts_ideal.get(s, 0) for s in all_states}
counts_noisy_full = {s: counts_noisy.get(s, 0) for s in all_states}

plot_histogram([counts_ideal_full, counts_noisy_full], 
               legend=['Ideal (Vector Estocástico Matemático)', 'Realidad (Decoherencia y Ruido ibmqx4)'], 
               color=['steelblue', 'coral'], 
               title='Reversión del Tiempo: Teoría vs Termodinámica del Hardware',
               ax=ax)

plt.tight_layout()
fig.savefig('qiskit_reality_noise.png', dpi=150)
print("¡Diagrama comparativo generado en qiskit_reality_noise.png!")
