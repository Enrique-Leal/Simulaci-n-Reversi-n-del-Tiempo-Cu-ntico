use rustfft::{FftPlanner, FftDirection};
use num_complex::Complex;
use image::{ImageBuffer, Rgb};
use rand::Rng;
use rand_distr::{Normal, Distribution};
use rayon::prelude::*;
use std::f64::consts::PI;
use std::fs;

// Parámetros Físicos
const N: usize = 256;
const L: f64 = 20.0;
const DX: f64 = L / N as f64;
const DT: f64 = 0.002;
const GAMMA: f64 = 0.05; // Factor Lindblad

// Animación
const SUBSTEPS_PER_FRAME: usize = 10;
const FRAMES_PHASE: usize = 100; // Total pasos = FRAMES * SUBSTEPS * DT = 100 * 10 * 0.002 = 2.0 seg

fn fftfreq(n: usize, d: f64) -> Vec<f64> {
    let mut freqs = vec![0.0; n];
    let val = 1.0 / (n as f64 * d);
    for i in 0..n {
        let k = if i < (n + 1) / 2 { i as i32 } else { i as i32 - n as i32 };
        freqs[i] = k as f64 * val * 2.0 * PI;
    }
    freqs
}

fn fft2d(buffer: &mut [Complex<f64>], direction: FftDirection) {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft(N, direction);

    // Filas
    buffer.par_chunks_mut(N).for_each(|row| {
        fft.process(row);
    });

    // Columnas
    // Transponer, FFT, Transponer
    let mut transposed = vec![Complex::new(0.0, 0.0); N * N];
    for row in 0..N {
        for col in 0..N {
            transposed[col * N + row] = buffer[row * N + col];
        }
    }
    
    transposed.par_chunks_mut(N).for_each(|col| {
        fft.process(col);
    });

    for row in 0..N {
        for col in 0..N {
            buffer[row * N + col] = transposed[col * N + row];
        }
    }

    if direction == FftDirection::Inverse {
        let norm = (N * N) as f64;
        buffer.par_iter_mut().for_each(|x| *x = *x / norm);
    }
}

fn hsv_to_rgb(h: f64, s: f64, v: f64) -> [u8; 3] {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);

    let (r, g, b) = match i as i32 % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [(r * 255.0).min(255.0) as u8, (g * 255.0).min(255.0) as u8, (b * 255.0).min(255.0) as u8]
}

fn save_frame(buffer: &[Complex<f64>], frame_idx: usize) {
    let mut img = ImageBuffer::new(N as u32, N as u32);
    
    // Calcular densidad máxima global (aprox) para normalizar visualmente el brillo
    let mut rho_max = 0.0_f64;
    for z in buffer.iter() {
        let rho = z.norm_sqr();
        if rho > rho_max { rho_max = rho; }
    }
    if rho_max == 0.0 { rho_max = 1.0; }

    for y in 0..N {
        for x in 0..N {
            let z = buffer[y * N + x];
            let rho = z.norm_sqr();
            let phase = z.arg();

            let mut h = (phase + PI) / (2.0 * PI);
            h = h.rem_euclid(1.0);
            
            // Gamma correction para visibilidad del vacío y neblina cuántica
            let v = (rho / rho_max).powf(0.5).min(1.0); 
            
            let rgb = hsv_to_rgb(h, 1.0, v);
            img.put_pixel(x as u32, y as u32, Rgb(rgb));
        }
    }
    img.save(format!("frames/frame_{:04}.png", frame_idx)).unwrap();
}

fn main() {
    fs::create_dir_all("frames").unwrap();
    
    let mut psi = vec![Complex::new(0.0, 0.0); N * N];
    let sigma = 1.0;
    let norm = 1.0 / (sigma * (PI).sqrt());

    for row in 0..N {
        for col in 0..N {
            let x = (col as f64 - N as f64 / 2.0) * DX;
            let y = (row as f64 - N as f64 / 2.0) * DX;
            let envelope = (-(x*x + y*y) / (2.0 * sigma * sigma)).exp();
            psi[row * N + col] = Complex::new(norm * envelope, 0.0);
        }
    }

    let freqs = fftfreq(N, DX);
    let mut k_phase = vec![Complex::new(0.0, 0.0); N * N];
    for row in 0..N {
        for col in 0..N {
            let kx = freqs[col];
            let ky = freqs[row];
            let phase = -0.5 * (kx*kx + ky*ky) * DT;
            k_phase[row * N + col] = Complex::new(0.0, phase).exp();
        }
    }

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut frame_count = 0;

    println!("Iniciando Engine Rust de Alta Precisión Cuántica...");
    println!("Resolución: {}x{}, Paso Temporal dt = {}", N, N, DT);
    
    // FASE 1: Dispersión Termodinámica
    for step in 0..FRAMES_PHASE {
        save_frame(&psi, frame_count);
        frame_count += 1;
        
        for _ in 0..SUBSTEPS_PER_FRAME {
            // Schrodinger Evol
            fft2d(&mut psi, FftDirection::Forward);
            psi.par_iter_mut().zip(&k_phase).for_each(|(z, &k)| *z = *z * k);
            fft2d(&mut psi, FftDirection::Inverse);

            // Lindblad Decoherence
            let mut rng = rand::thread_rng();
            let mut sq_norm = 0.0;
            for z in psi.iter_mut() {
                let dW = Complex::new(normal.sample(&mut rng), normal.sample(&mut rng)) * DT.sqrt();
                let damp = (-GAMMA * DT).exp();
                let phase_noise = (Complex::new(0.0, 1.0) * GAMMA.sqrt() * dW).exp();
                *z = *z * damp * phase_noise;
                sq_norm += z.norm_sqr();
            }
            
            // Re-normalizar
            let norm_factor = 1.0 / (sq_norm * DX * DX).sqrt();
            psi.par_iter_mut().for_each(|z| *z = *z * norm_factor);
        }
        if step % 10 == 0 { println!("Progreso Adelante: {} / {}", step, FRAMES_PHASE); }
    }

    // FASE 2: Conjugación (Pulsos de Reversión)
    println!("Aplicando Conjugación K...");
    for z in psi.iter_mut() {
        *z = z.conj();
    }
    
    for _ in 0..10 { // Pausa visual
        save_frame(&psi, frame_count);
        frame_count += 1;
    }

    // FASE 3: Reversión bajo Entropía
    for step in 0..FRAMES_PHASE {
        save_frame(&psi, frame_count);
        frame_count += 1;
        
        for _ in 0..SUBSTEPS_PER_FRAME {
            // Schrodinger Evol
            fft2d(&mut psi, FftDirection::Forward);
            psi.par_iter_mut().zip(&k_phase).for_each(|(z, &k)| *z = *z * k);
            fft2d(&mut psi, FftDirection::Inverse);

            // Lindblad Decoherence (La entropía sigue avanzando mientras el tiempo de fase retrocede)
            let mut rng = rand::thread_rng();
            let mut sq_norm = 0.0;
            for z in psi.iter_mut() {
                let dW = Complex::new(normal.sample(&mut rng), normal.sample(&mut rng)) * DT.sqrt();
                let damp = (-GAMMA * DT).exp();
                let phase_noise = (Complex::new(0.0, 1.0) * GAMMA.sqrt() * dW).exp();
                *z = *z * damp * phase_noise;
                sq_norm += z.norm_sqr();
            }
            let norm_factor = 1.0 / (sq_norm * DX * DX).sqrt();
            psi.par_iter_mut().for_each(|z| *z = *z * norm_factor);
        }
        if step % 10 == 0 { println!("Progreso Reversión: {} / {}", step, FRAMES_PHASE); }
    }

    for _ in 0..20 {
        save_frame(&psi, frame_count);
        frame_count += 1;
    }

    println!("¡Generación de Frames Cuánticos Exitosa! Ejecuta ffmpeg para empaquetar.");
}
