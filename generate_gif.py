import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ============================================================
# 1. FUNCIONES DEL PAQUETE DE ONDA (del paper)
# ============================================================

def wave_packet(x, tau, sigma0=1.0, m=1.0, hbar=1.0):
    eps = 1e-10
    sigma_t = sigma0 * np.sqrt(1 + (hbar * tau / (m * sigma0**2))**2)
    phase   = (m * x**2) / (2 * hbar * (tau + eps))
    norm    = 1.0 / (np.sqrt(2 * np.pi) * sigma_t)
    amplitude = norm * np.exp(-x**2 / (2 * sigma_t**2))
    return amplitude * np.exp(1j * phase)

def time_reverse_wavepacket(psi):
    return np.conj(psi)

def evolve_from_state(psi_init, x, dtau, m=1.0, hbar=1.0):
    """
    Evolución temporal simple en el espacio de momentos vía FFT.
    """
    dx = x[1] - x[0]
    k  = np.fft.fftfreq(len(x), d=dx/(2*np.pi))
    
    # Transformada al espacio de momentos
    psi_k = np.fft.fft(psi_init)
    
    # Evolución libre exp(-i P^2 / (2m*hbar) * dtau)
    # Aquí tau está en unidades relativas, ajustamos el factor
    phase = - (hbar * k**2 / (2*m)) * dtau
    psi_k_evolved = psi_k * np.exp(1j * phase)
    
    # Transformada inversa al espacio de posición
    psi_x_evolved = np.fft.ifft(psi_k_evolved)
    return psi_x_evolved

# ============================================================
# 2. CONFIGURACIÓN DE LA ANIMALCIÓN
# ============================================================

x      = np.linspace(-15, 15, 1000)
tau_max = 3.0
sigma0 = 1.0
frames = 120  # Total frames

# Fases de la animación:
# 1. (0 to frames//2) Evolución forward: tau = 0 to tau_max
# 2. (frames//2) Reversión instantánea (Conjugación)
# 3. (frames//2 to frames) Evolución hacia atrás (re-focalización)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-15, 15)
ax.set_ylim(0, 0.5)
ax.set_xlabel('x (u.a.)')
ax.set_ylabel(r'$|\Psi|^2$')
ax.grid(True, alpha=0.3)
line, = ax.plot([], [], 'b-', lw=3)
title = ax.set_title('')

# Generar la secuencia de tiempos
taus_forward  = np.linspace(1e-8, tau_max, frames // 2)

# Pre-calcular todos los estados
psis = []

# Phase 1: Forward evolution
for t in taus_forward:
    psis.append(wave_packet(x, tau=t, sigma0=sigma0))

# The actual reversed state at tau_max
psi_tau_max = psis[-1]
psi_rev = time_reverse_wavepacket(psi_tau_max)

# Phase 2: Reverse evolution (we use the same FFT evolution to show actual propagation)
taus_backward = taus_forward[::-1]
dtau_step = taus_forward[1] - taus_forward[0]

current_psi = psi_rev
for t in taus_backward:
    psis.append(current_psi)
    current_psi = evolve_from_state(current_psi, x, dtau=dtau_step)

def init():
    line.set_data([], [])
    return (line,)

def animate(i):
    psi = psis[i]
    density = np.abs(psi)**2
    # Normalize density for visualization to keep it comparable
    dx = x[1] - x[0]
    norm = np.trapz(density, x)
    density = density / norm
    
    line.set_data(x, density)
    
    if i < frames // 2:
        line.set_color('blue')
        t_val = taus_forward[i]
        title.set_text(rf'Evolución Dispersiva Forward $\tau$ = {t_val:.2f}')
    elif i == frames // 2:
         line.set_color('green')
         line.set_linestyle('--')
         title.set_text('¡REVERSIÓN! (Conjugación Compleja $\Psi \rightarrow \Psi^*$)')
    else:
        line.set_color('red')
        line.set_linestyle('-')
        # Time remaining to reach focus
        t_val = taus_backward[i - frames//2]
        title.set_text(rf'Re-focalización Temporal $\tau_{{faltante}}$ = {t_val:.2f}')
        
    return (line,)

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(psis), interval=50, blit=True)

print("Generando GIF, por favor espera...")
anim.save('wave_packet_reversal.gif', writer='pillow', fps=20)
print("¡GIF guardado como wave_packet_reversal.gif!")
