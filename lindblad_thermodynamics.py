import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb

# ============================================================
# 1. PARÁMETROS FÍSICOS DE LA SIMULACIÓN 2D
# ============================================================
N = 120          # Resolución de la cuadrícula
L = 20.0         # Tamaño físico del pozo
dx = L / N
x = np.linspace(-L/2, L/2, N, endpoint=False)
y = np.linspace(-L/2, L/2, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# Frecuencias espaciales (momento k)
kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky)

# Tiempos
tau_max = 2.0
frames_forward = 40
dt = tau_max / frames_forward

# PARÁMETRO DE DECOHERENCIA (LINDBLAD)
# Gamma controla qué tan rápido el ambiente "mira" e interacciona con el electrón
# Si Gamma = 0, es la simulación de Schrödinger pura perfecta.
# Si Gamma > 0, es el universo real destruyendo la fase cuántica.
GAMMA = 0.05 

# ============================================================
# 2. FUNCIONES DE ONDA Y EVOLUCIÓN ESTOCÁSTICA DE SCHRÖDINGER
# ============================================================
def init_wave_packet(sigma=1.0, px=0.0, py=0.0):
    norm = 1.0 / (sigma * np.sqrt(np.pi))
    envelope = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    phase = np.exp(1j * (px * X + py * Y))
    return norm * envelope * phase

def evolve_schrodinger(psi, dt):
    """Evolución Unitaria pura (Aislada)"""
    psi_k = np.fft.fft2(psi)
    kinetic_phase = np.exp(-1j * (KX**2 + KY**2) * dt / 2.0)
    return np.fft.ifft2(psi_k * kinetic_phase)

def apply_lindblad_decoherence(psi, gamma, dt):
    """
    APROXIMACIÓN DE LA ECUACIÓN DE LINDBLAD (Quantum State Diffusion)
    En lugar de la costosa matriz de densidad volumétrica (N^4), usamos 
    una trayectoria de difusión estocástica. El ambiente colapsa aleatoriamente 
    la fase local introduciendo "ruido blanco cuántico" proporcional a dt y Gamma.
    Este es el análogo espacial de T2 (De-phasing continuo).
    """
    # Ruido Gaussiano complejo (Fluctuaciones térmicas del vacío)
    noise_real = np.random.normal(0, 1, (N, N))
    noise_imag = np.random.normal(0, 1, (N, N))
    dW = (noise_real + 1j * noise_imag) * np.sqrt(dt)
    
    # El operador de disipación deforma la fase y amplitud levemente
    dissipation = np.exp(-gamma * dt) * np.exp(1j * np.sqrt(gamma) * dW)
    
    psi_noisy = psi * dissipation
    
    # Renormalizar porque el sistema sigue existiendo (conservación de la probabilidad local)
    norm = np.sqrt(np.sum(np.abs(psi_noisy)**2) * dx * dx)
    return psi_noisy / norm

def complex_to_rgba(Z):
    rho = np.abs(Z)**2
    phase = np.angle(Z)
    
    H = (phase + np.pi) / (2 * np.pi)
    rho_max = np.max(rho)
    V = (rho / rho_max)**0.7 if rho_max > 0 else np.zeros_like(rho)
    S = np.ones_like(H)
    
    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)
    return np.dstack((RGB, np.ones_like(V)))

# ============================================================
# 3. GENERACIÓN DEL LABORATORIO ABIERTO (ENTROPÍA VIVA)
# ============================================================
print("Generando laboratorio atómico 2D con Termodinámica de Lindblad...")
print(f"Factor de Decoherencia Ambiental: Gamma = {GAMMA}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor='#050505')
fig.suptitle("Dinámica de Sistemas Abiertos:\nDisipación Térmica (Lindblad) vs Reversión Temporal", color='white', fontsize=14)

for ax in [ax1, ax2]:
    ax.set_facecolor('black')
    ax.set_xticks([])
    ax.set_yticks([])

ax1.set_title("Nube Colorimétrica (El Calor rompe la Fase)", color='white', fontsize=11)
ax2.set_title("Densidad de Probabilidad (Re-enfoque Incompleto)", color='white', fontsize=11)

psi = init_wave_packet()
img_rgba = ax1.imshow(complex_to_rgba(psi), origin='lower', extent=[-L/2, L/2, -L/2, L/2])
img_dens = ax2.imshow(np.abs(psi)**2, origin='lower', extent=[-L/2, L/2, -L/2, L/2], cmap='magma')

text_status = fig.text(0.5, 0.05, '', color='white', ha='center', fontsize=12, 
                       bbox=dict(facecolor='red', alpha=0.3, boxstyle='round,pad=0.5'))

psis_frames = []
estados = []

# SIMULACIÓN (DIFUSIÓN ESTOCÁSTICA CONTINUA)
current_psi = psi
for i in range(frames_forward):
    psis_frames.append(current_psi)
    estados.append(f"Flecha del Tiempo 1: Dispersión + Calor Ambiental - t={i*dt:.2f}")
    
    # 1. Evolución de la Física Cuántica (Aislada)
    current_psi = evolve_schrodinger(current_psi, dt)
    # 2. Intervención de la Termodinámica (Abierta)
    current_psi = apply_lindblad_decoherence(current_psi, GAMMA, dt)

# CONJUGACIÓN TLI
psis_frames.append(current_psi)
estados.append("IMPACTO TLI: Conjugación K Intentando Revertir el Caos...")
reversed_psi = np.conj(current_psi)

for _ in range(5):
    psis_frames.append(reversed_psi)
    estados.append("IMPACTO TLI: Reversión Óptica Activada")

# RE-FOCALIZACIÓN BAJO CALOR
current_psi = reversed_psi
for i in range(frames_forward):
    psis_frames.append(current_psi)
    estados.append(f"Flecha del Tiempo 2: Re-focalización luchando contra la Entropía - t={(frames_forward-i)*dt:.2f}")
    
    current_psi = evolve_schrodinger(current_psi, dt)
    current_psi = apply_lindblad_decoherence(current_psi, GAMMA, dt)

for _ in range(12):
    psis_frames.append(current_psi)
    estados.append("Resultado: Falla Termodinámica. Nube Irreversiblemente Dañada.")
    
# METRICAS DEL FRACASO:
fid_final = np.abs(np.sum(np.conj(psi) * current_psi) * dx * dx)**2
print(f"Fidelidad final Post-Reversión con Calor (Gamma={GAMMA}): {fid_final*100:.2f}%")

def animate(frame_idx):
    if frame_idx >= len(psis_frames): return [img_rgba, img_dens, text_status]
    psi_frame = psis_frames[frame_idx]
    
    img_rgba.set_array(complex_to_rgba(psi_frame))
    density = np.abs(psi_frame)**2
    img_dens.set_array(density / (np.max(density)+1e-10)) # Mapeo dinámico robusto
    text_status.set_text(estados[frame_idx])
    
    return [img_rgba, img_dens, text_status]

anim = animation.FuncAnimation(fig, animate, frames=len(psis_frames), interval=60, blit=True)
filename = "lindblad_thermodynamic_reversal.gif"
anim.save(filename, writer='pillow', fps=15)
print(f"¡Animación Termodinámica Guardada en {filename}!")
