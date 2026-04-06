//! Competitive learning network for palette generation.
//!
//! Trains a 1D Kohonen self-organizing map (SOM) on the **spectral color
//! transform** of the image rather than raw pixel data. Each frequency bin
//! (u, v) in the 2D FFT has an amplitude per Lab channel — the triplet
//! (|L̂|, |â|, |b̂|) is treated as a "spectral color" describing what color
//! content exists at that frequency. Training the network on these spectral
//! colors gives it a view of the color space that naturally separates
//! highlight/detail colors from bulk colors, similar to how bit-bucketing
//! partitions color space but driven by frequency structure.
//!
//! The resulting palette can be used on its own or combined with a
//! pixel-trained palette so large uniform regions still get adequate
//! representation.

use image::Rgba;
use image::RgbaImage;
use palette::Lab;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;
use rayon::slice::ParallelSliceMut;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

use crate::PaletteBuilder;
use crate::private;
use crate::rgba_to_lab;

const TRAINING_PASSES: usize = 8;
const ALPHA_START: f32 = 0.8;
const ALPHA_END: f32 = 0.01;
const RADIUS_FRACTION: f32 = 0.25;

pub struct FocalPaletteBuilder;

impl private::Sealed for FocalPaletteBuilder {}

impl PaletteBuilder for FocalPaletteBuilder {
    const NAME: &'static str = "Focal";

    fn build_palette(
        image: &RgbaImage,
        palette_size: usize,
    ) -> Vec<Lab> {
        let width = image.width();
        let height = image.height();
        let raw = image.to_vec();

        let n_pixels = width as usize * height as usize;
        let mut pixels = vec![<Lab>::new(0.0, 0.0, 0.0); n_pixels];

        raw.par_chunks(4).zip(&mut pixels).for_each(|(p, dest)| {
            *dest = rgba_to_lab(Rgba([p[0], p[1], p[2], p[3]]));
        });

        let spectral_colors = spectral_color_transform(&pixels, width, height);

        train(&spectral_colors, &pixels, palette_size)
    }
}

#[derive(Clone, Copy, Debug)]
struct SpectralPixel {
    l: f32,
    a: f32,
    b: f32,
}

impl std::ops::Div<f32> for SpectralPixel {
    type Output = Self;

    fn div(
        self,
        d: f32,
    ) -> Self {
        Self {
            l: self.l / d,
            a: self.a / d,
            b: self.b / d,
        }
    }
}

impl std::ops::Add for SpectralPixel {
    type Output = Self;

    fn add(
        self,
        rhs: Self,
    ) -> Self {
        Self {
            l: self.l + rhs.l,
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}

pub(crate) fn fft_2d(
    buffer: &mut [Complex<f32>],
    scratch: &mut Vec<Complex<f32>>,
    width: usize,
    height: usize,
    fft_row: &dyn rustfft::Fft<f32>,
    fft_col: &dyn rustfft::Fft<f32>,
) {
    scratch.resize(buffer.len(), Complex::zero());

    buffer.par_chunks_mut(width).for_each(|chunk| {
        fft_row.process(chunk);
    });

    (0..width)
        .into_par_iter()
        .zip(scratch.par_chunks_mut(height))
        .for_each(|(x, col)| {
            for y in 0..height {
                col[y] = buffer[y * width + x];
            }
        });

    scratch.par_chunks_mut(height).for_each(|chunk| {
        fft_col.process(chunk);
    });

    (0..height)
        .into_par_iter()
        .zip(buffer.par_chunks_mut(width))
        .for_each(|(y, row)| {
            for x in 0..width {
                row[x] = scratch[x * height + y];
            }
        });
}



fn spectral_color_transform(
    pixels: &[Lab],
    image_width: u32,
    image_height: u32,
) -> Vec<SpectralPixel> {
    let wp2 = image_width.next_power_of_two() as usize;
    let hp2 = image_height.next_power_of_two() as usize;
    let n = wp2 * hp2;

    let mut planner = FftPlanner::new();
    let fft_row = planner.plan_fft_forward(wp2);
    let fft_col = planner.plan_fft_forward(hp2);

    let mut l_buf = vec![Complex::zero(); n];
    let mut a_buf = vec![Complex::zero(); n];
    let mut b_buf = vec![Complex::zero(); n];

    for y in 0..image_height as usize {
        for x in 0..image_width as usize {
            let px = pixels[y * image_width as usize + x];
            l_buf[y * wp2 + x] = Complex { re: px.l, im: 0.0 };
            a_buf[y * wp2 + x] = Complex { re: px.a, im: 0.0 };
            b_buf[y * wp2 + x] = Complex { re: px.b, im: 0.0 };
        }
    }

    let mut scratch = Vec::new();

    fft_2d(
        &mut l_buf,
        &mut scratch,
        wp2,
        hp2,
        fft_row.as_ref(),
        fft_col.as_ref(),
    );
    fft_2d(
        &mut a_buf,
        &mut scratch,
        wp2,
        hp2,
        fft_row.as_ref(),
        fft_col.as_ref(),
    );
    fft_2d(
        &mut b_buf,
        &mut scratch,
        wp2,
        hp2,
        fft_row.as_ref(),
        fft_col.as_ref(),
    );

    let l_sr = amplify(&mut planner, &mut scratch, &l_buf, wp2, hp2);
    let a_sr = amplify(&mut planner, &mut scratch, &a_buf, wp2, hp2);
    let b_sr = amplify(&mut planner, &mut scratch, &b_buf, wp2, hp2);

    let w = image_width as usize;
    let h = image_height as usize;
    let mut result = Vec::with_capacity(w * h);

    for y in 0..h {
        for x in 0..w {
            let i = y * wp2 + x;
            result.push(SpectralPixel {
                l: l_sr[i],
                a: a_sr[i],
                b: b_sr[i],
            });
        }
    }

    result
}

fn amplify(
    planner: &mut FftPlanner<f32>,
    scratch: &mut Vec<Complex<f32>>,
    fft_buf: &[Complex<f32>],
    wp2: usize,
    hp2: usize,
) -> Vec<f32> {
    let n = wp2 * hp2;

    // log-amplitude and phase
    let mut amp_amp = vec![0.0f32; n];
    let mut phase = vec![0.0f32; n];
    fft_buf
        .par_iter()
        .copied()
        .zip(&mut amp_amp)
        .for_each(|(c, dest)| *dest = c.norm().max(f32::EPSILON).ln());
    fft_buf
        .par_iter()
        .copied()
        .zip(&mut phase)
        .for_each(|(c, dest)| *dest = c.arg());

    let amp_mean = amp_amp.par_iter().copied().sum::<f32>() / n as f32;

    let mut buffer = vec![Complex::zero(); n];
    amp_amp
        .into_par_iter()
        .zip(phase.into_par_iter())
        .zip(&mut buffer)
        .for_each(|((a, p), dest)| {
            let diff = a - amp_mean;
            *dest = Complex {
                re: diff.powi(3) * p.cos(),
                im: diff.powi(3) * p.sin(),
            };
        });

    let ifft_row = planner.plan_fft_inverse(wp2);
    let ifft_col = planner.plan_fft_inverse(hp2);

    fft_2d(
        &mut buffer,
        scratch,
        wp2,
        hp2,
        ifft_row.as_ref(),
        ifft_col.as_ref(),
    );

    let mut out = vec![0.0f32; n];
    buffer
        .par_iter()
        .copied()
        .zip(&mut out)
        .for_each(|(c, dest)| *dest = c.norm());

    out
}

fn spectral_to_lab_via_nearest(
    neurons: &[SpectralPixel],
    spectral_pixels: &[SpectralPixel],
    lab_pixels: &[Lab],
) -> Vec<Lab> {
    neurons
        .iter()
        .map(|neuron| {
            let best = spectral_pixels
                .iter()
                .enumerate()
                .map(|(i, sp)| {
                    let dl = neuron.l - sp.l;
                    let da = neuron.a - sp.a;
                    let db = neuron.b - sp.b;
                    (i, dl * dl + da * da + db * db)
                })
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .unwrap_or(0);
            lab_pixels[best]
        })
        .collect()
}

fn maximin_init(
    pixels: &[SpectralPixel],
    n: usize,
) -> Vec<[f32; 3]> {
    let mut neurons = Vec::with_capacity(n);

    let centroid = pixels.par_iter().copied().reduce(
        || SpectralPixel {
            l: 0.0,
            a: 0.0,
            b: 0.0,
        },
        |a, b| a + b,
    ) / pixels.len() as f32;

    let first = pixels
        .par_iter()
        .enumerate()
        .map(|(i, px)| {
            let dl = px.l - centroid.l;
            let da = px.a - centroid.a;
            let db = px.b - centroid.b;
            (i, dl * dl + da * da + db * db)
        })
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let p = pixels[first];
    neurons.push([p.l, p.a, p.b]);

    for _ in 1..n {
        let best = pixels
            .par_iter()
            .enumerate()
            .map(|(idx, px)| {
                let min_d = neurons
                    .iter()
                    .map(|n| {
                        let dl = px.l - n[0];
                        let da = px.a - n[1];
                        let db = px.b - n[2];
                        dl * dl + da * da + db * db
                    })
                    .fold(f32::MAX, f32::min);
                (idx, min_d)
            })
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let p = pixels[best];
        neurons.push([p.l, p.a, p.b]);
    }

    neurons
}

fn train(
    spectral_pixels: &[SpectralPixel],
    lab_pixels: &[Lab],
    palette_size: usize,
) -> Vec<Lab> {
    let pixels = spectral_pixels;

    if pixels.is_empty() {
        return vec![Lab::new(50.0, 0.0, 0.0)];
    }

    let n = palette_size;
    let mut neurons = maximin_init(pixels, n);

    let total_samples = pixels.len() * TRAINING_PASSES;
    let radius_start = (n as f32 * RADIUS_FRACTION).max(1.0);
    let radius_end = 0.5f32;

    let golden_ratio = 0.6180339887_f64;
    let len = pixels.len();
    let mut probe = 0.0_f64;

    for step in 0..total_samples {
        let t = step as f32 / total_samples as f32;

        let alpha = ALPHA_START * (ALPHA_END / ALPHA_START).powf(t);
        let radius = radius_start * (radius_end / radius_start).powf(t);
        let radius_sq = radius * radius;

        probe += golden_ratio;
        if probe >= 1.0 {
            probe -= 1.0;
        }
        let px_idx = (probe * len as f64) as usize;
        let input = pixels[px_idx];

        let bmu = find_bmu(&neurons, &input);

        let lo = if bmu as f32 - radius < 0.0 {
            0
        } else {
            (bmu as f32 - radius) as usize
        };
        let hi = ((bmu as f32 + radius) as usize + 1).min(n);

        for j in lo..hi {
            let dist = (j as f32 - bmu as f32) * (j as f32 - bmu as f32);
            let h = (-dist / (2.0 * radius_sq)).exp();
            let lr = alpha * h;

            neurons[j][0] += lr * (input.l - neurons[j][0]);
            neurons[j][1] += lr * (input.a - neurons[j][1]);
            neurons[j][2] += lr * (input.b - neurons[j][2]);
        }
    }

    let neuron_spectral: Vec<SpectralPixel> = neurons
        .iter()
        .map(|n| SpectralPixel {
            l: n[0],
            a: n[1],
            b: n[2],
        })
        .collect();

    spectral_to_lab_via_nearest(&neuron_spectral, pixels, lab_pixels)
}

fn find_bmu(
    neurons: &[[f32; 3]],
    input: &SpectralPixel,
) -> usize {
    let mut best = 0;
    let mut best_dist = f32::MAX;
    for (i, n) in neurons.iter().enumerate() {
        let dl = input.l - n[0];
        let da = input.a - n[1];
        let db = input.b - n[2];
        let d = dl * dl + da * da + db * db;
        if d < best_dist {
            best_dist = d;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_palette() {
        let mut img = RgbaImage::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                let color = if x < 16 {
                    Rgba([255, 0, 0, 255])
                } else {
                    Rgba([0, 0, 255, 255])
                };
                img.put_pixel(x, y, color);
            }
        }

        let palette = FocalPaletteBuilder::build_palette(&img, 8);
        assert!(!palette.is_empty());
        assert!(palette.len() <= 8);
    }

    #[test]
    fn single_color() {
        let img = RgbaImage::from_pixel(16, 16, Rgba([128, 128, 128, 255]));
        let palette = FocalPaletteBuilder::build_palette(&img, 4);
        assert!(!palette.is_empty());
    }
}
