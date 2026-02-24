import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb

# ============================================================
# 1. PARÁMETROS DE LA SIMULACIÓN
# ============================================================
# Dimensiones espaciales
N = 128          # Resolución de la grilla (NxN)
L = 20.0         # Tamaño de la caja
dx = L / N
x = np.linspace(-L/2, L/2, N, endpoint=False)
y = np.linspace(-L/2, L/2, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# Parámetros físicos (unidades atómicas simplificadas m=1, hbar=1)
sigma0 = 1.0     # Anchura inicial del paquete
p0_x = 0.0       # Momento inicial en x
p0_y = 0.0       # Momento inicial en y

# Tiempos
tau_max = 2.0    # Tiempo total de dispersión forward
frames_forward = 40
dtau = tau_max / frames_forward

# ============================================================
# 2. FUNCIONES DE ONDA Y EVOLUCIÓN (SPLIT-STEP FOURIER)
# ============================================================
def init_wave_packet(X, Y, sigma, px, py):
    """Crea un paquete de onda Gaussiano 2D."""
    norm = 1.0 / (sigma * np.sqrt(np.pi))
    envelope = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    phase = np.exp(1j * (px * X + py * Y))
    return norm * envelope * phase

def evolve_free_space(psi, kx, ky, dt):
    """Evolución paso de tiempo dt en el espacio de Fourier (momento)."""
    # Transformada al espacio K
    psi_k = np.fft.fft2(psi)
    
    # Fase de evolución libre exp(-i (kx^2 + ky^2) dt / 2)
    kinetic_phase = np.exp(-1j * (kx**2 + ky**2) * dt / 2.0)
    
    # Aplicar fase y volver al espacio real
    psi_k_evolved = psi_k * kinetic_phase
    return np.fft.ifft2(psi_k_evolved)

def complex_to_rgba(Z):
    """
    Mapeo de función de onda compleja a RGBA:
    - Brillo/Intensidad (Value) representa la densidad de probabilidad |Psi|^2
    - Color (Hue) representa la fase imaginaria arg(Psi)
    """
    # Densidad y fase
    rho = np.abs(Z)**2
    phase = np.angle(Z)
    
    # Normalizar fase a [0, 1] para Hue
    H = (phase + np.pi) / (2 * np.pi)
    
    # Normalizar densidad para Value (Intensidad visual)
    # Exageramos un poco los valores bajos para que se vea el esparcimiento
    rho_max = np.max(rho)
    if rho_max > 0:
        V = (rho / rho_max)**0.6  # Gamma correction para realzar la neblina
    else:
        V = np.zeros_like(rho)
        
    S = np.ones_like(H) # Saturación máxima
    
    # Apilar en HSV y convertir a RGB
    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)
    
    # Crear imagen RGBA (Alpha=Value para fondo oscuro)
    RGBA = np.dstack((RGB, np.ones_like(V)))
    return RGBA

# ============================================================
# 3. PREPARAR EL ENTORNO ESPECTRAL (K-Space)
# ============================================================
# Frecuencias espaciales (kx, ky)
kx1d = np.fft.fftfreq(N, d=dx) * 2 * np.pi
ky1d = np.fft.fftfreq(N, d=dx) * 2 * np.pi
KX, KY = np.meshgrid(kx1d, ky1d)

# ============================================================
# 4. BUCLE DE ANIMACIÓN
# ============================================================
print("Generando laboratorio atómico 2D...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor='black')
fig.suptitle("Simulación Física: Electrón y Reversión Temporal", color='white', fontsize=14)

for ax in [ax1, ax2]:
    ax.set_facecolor('black')
    ax.set_xticks([])
    ax.set_yticks([])

ax1.set_title("Densidad y Fase Colorimétrica", color='white')
ax2.set_title("Densidad de Probabilidad 3D (Corte Superior)", color='white')

# Estado inicial
psi = init_wave_packet(X, Y, sigma0, p0_x, p0_y)

# Imagen RGB
img_rgba = ax1.imshow(complex_to_rgba(psi), origin='lower', extent=[-L/2, L/2, -L/2, L/2])

# Superficie 3D (plotteada como imshow con colormap inferno para la otra vista)
img_dens = ax2.imshow(np.abs(psi)**2, origin='lower', extent=[-L/2, L/2, -L/2, L/2], cmap='inferno')

text_status = fig.text(0.5, 0.05, '', color='white', ha='center', fontsize=12, 
                       bbox=dict(facecolor='white', alpha=0.1, boxstyle='round,pad=0.5'))

# Pre-computar todos los frames para guardar el GIF fluidamente
psis_frames = []
estados = [] # Guarda strings de estado para el texto

# FASE 1: Dispersión Termodinámica (Adquirir Entropía/Ensanchamiento)
current_psi = psi
for i in range(frames_forward):
    psis_frames.append(current_psi)
    estados.append(f"Fase 1: Dispersión Natural (Forward) - t={i*dtau:.2f}")
    current_psi = evolve_free_space(current_psi, KX, KY, dtau)

# FASE 2: Conjugación (El Pulso del Láser TLI)
psis_frames.append(current_psi)
estados.append("Fase 2: IMPACTO TLI -> Conjugación Compleja (K)")
reversed_psi = np.conj(current_psi)

# Pausa visual mostrando el pulso flash
for _ in range(5):
    psis_frames.append(reversed_psi)
    estados.append("Fase 2: IMPACTO TLI -> Conjugación Compleja (K) Inviertiendo Fases!")

# FASE 3: Re-focalización (Rompiendo la flecha del tiempo)
# La evolución libre del estado conjugado lo comprime
current_psi = reversed_psi
for i in range(frames_forward):
    psis_frames.append(current_psi)
    estados.append(f"Fase 3: Re-focalización (Anti-Dispersión) - t={(frames_forward-i)*dtau:.2f} restante")
    current_psi = evolve_free_space(current_psi, KX, KY, dtau)

# Pausa visual con el electrón re-comprimido
for _ in range(10):
    psis_frames.append(current_psi)
    estados.append("Electrón Restaurado Exitosamente a su Estado Inicial")

def animate(frame_idx):
    psi_frame = psis_frames[frame_idx]
    
    # Actualizar RGBA
    rgba = complex_to_rgba(psi_frame)
    img_rgba.set_array(rgba)
    
    # Actualizar Densidad Infernal
    density = np.abs(psi_frame)**2
    # Normalizar para visualización consistente
    img_dens.set_array(density / np.max(density))
    
    # Actualizar Texto
    text_status.set_text(estados[frame_idx])
    
    return [img_rgba, img_dens, text_status]

anim = animation.FuncAnimation(fig, animate, frames=len(psis_frames), interval=60, blit=True)

filename = 'atomic_2d_reversal.gif'
print(f"Renderizando la simulación 2D en {filename}... (esto tomará unos segundos)")
anim.save(filename, writer='pillow', fps=15)
print("¡Renderizado Exitoso!")
