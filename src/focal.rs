//! Spectral-residual peak isolation for palette generation.
//!
//! Computes the spectral residual of the image in Lab color space, producing a
//! per-pixel saliency map. Peaks above a threshold (mean + 1.36 standard
//! deviation) are collected, then a greedy farthest-point algorithm selects
//! the most isolated peaks in Lab space, breaking ties by saliency. This
//! gives small palettes excellent coverage of visually distinct, salient
//! colors.

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

        isolate_peaks(&spectral_colors, &pixels, palette_size)
    }
}

#[derive(Clone, Copy, Debug)]
struct SpectralPixel {
    l: f32,
    a: f32,
    b: f32,
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

    let mut log_amp = vec![0.0f32; n];
    let mut phase = vec![0.0f32; n];
    fft_buf
        .par_iter()
        .copied()
        .zip(&mut log_amp)
        .for_each(|(c, dest)| *dest = c.norm().max(f32::EPSILON).ln());
    fft_buf
        .par_iter()
        .copied()
        .zip(&mut phase)
        .for_each(|(c, dest)| *dest = c.arg());

    let amp_mean = log_amp.par_iter().copied().sum::<f32>() / n as f32;

    let mut buffer = vec![Complex::zero(); n];
    log_amp
        .into_par_iter()
        .zip(phase.into_par_iter())
        .zip(&mut buffer)
        .for_each(|((a, p), dest)| {
            let diff = a - amp_mean;
            *dest = Complex {
                re: diff * p.cos(),
                im: diff * p.sin(),
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

fn isolate_peaks(
    spectral_pixels: &[SpectralPixel],
    lab_pixels: &[Lab],
    palette_size: usize,
) -> Vec<Lab> {
    if spectral_pixels.is_empty() {
        return vec![Lab::new(50.0, 0.0, 0.0)];
    }

    let saliency: Vec<f32> = spectral_pixels
        .par_iter()
        .map(|sp| (sp.l * sp.l + sp.a * sp.a + sp.b * sp.b).sqrt())
        .collect();

    let n = saliency.len() as f32;
    let mean = saliency.par_iter().copied().sum::<f32>() / n;
    let variance = saliency
        .par_iter()
        .map(|&s| (s - mean) * (s - mean))
        .sum::<f32>()
        / n;
    let threshold = mean + variance.sqrt() * 1.36;

    let mut peaks: Vec<(usize, f32)> = saliency
        .iter()
        .enumerate()
        .filter(|&(_, &s)| s > threshold)
        .map(|(i, &s)| (i, s))
        .collect();

    if peaks.len() < palette_size {
        let mut indexed: Vec<(usize, f32)> = saliency.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|(_, a), (_, b)| b.total_cmp(a));
        indexed.truncate(palette_size);
        peaks = indexed;
    }

    peaks.sort_unstable_by(|(_, a), (_, b)| b.total_cmp(a));

    farthest_point_selection(&peaks, lab_pixels, palette_size)
}

fn farthest_point_selection(
    peaks: &[(usize, f32)],
    lab_pixels: &[Lab],
    n: usize,
) -> Vec<Lab> {
    let mut selected: Vec<Lab> = Vec::with_capacity(n);
    let mut chosen = vec![false; peaks.len()];

    selected.push(lab_pixels[peaks[0].0]);
    chosen[0] = true;

    let mut min_dist: Vec<f32> = vec![f32::MAX; peaks.len()];

    for _ in 1..n.min(peaks.len()) {
        let last = selected.last().unwrap();

        for (j, (idx, _)) in peaks.iter().enumerate() {
            let lab = lab_pixels[*idx];
            let dl = lab.l - last.l;
            let da = lab.a - last.a;
            let db = lab.b - last.b;
            let d = dl * dl + da * da + db * db;
            if d < min_dist[j] {
                min_dist[j] = d;
            }
        }

        let best = peaks
            .iter()
            .enumerate()
            .filter(|(j, _)| !chosen[*j])
            .max_by(|(i, (_, sal_a)), (j, (_, sal_b))| {
                min_dist[*i]
                    .total_cmp(&min_dist[*j])
                    .then(sal_a.total_cmp(sal_b))
            });

        match best {
            Some((j, (idx, _))) => {
                selected.push(lab_pixels[*idx]);
                chosen[j] = true;
            }
            None => break,
        }
    }

    selected
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
