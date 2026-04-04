//! Use weighted pixels based on the image's spectral properties to provided
//! weighted input to k-means.
//!
//! Weights are computed using the following steps:
//! 1. Compute the saliency maps for each channel (L, a, b) using a combination
//!    of the following:
//!    - Spectral residual is the difference between the blurred natural log of
//!      the amplitude calculated by the FFT and the natural log of the
//!      amplitude reconstructed from the inverse FFT using the original phase
//!      spectrum.
//!    - Modulated spectral residual is the same as the above, but the amplitude
//!      and phase have flattened mid values.
//!    - Phase spectrum is the phase of the FFT.
//!    - Amplitude spectrum is the amplitude of the FFT modulated to have a
//!      flattened midrange.
//! 2. Compute the average distance of each pixel to the centroid of the image
//!    transformed to the range \[0, 1].
//! 3. Compute the average local distance of each pixel to its neighbors within
//!    a window of size ln(pixels.len()) / 2. This is used to help identify
//!    edges and features in the image.
//! 4. Compute the final weight for each pixel as a combination of the saliency
//!    maps and the local distance: lerp(a, sr.max(mod_sr), p, ld) where sr is
//!    the spectral residual, mod_sr is the modulated spectral residual, p is
//!    the phase spectrum, a is the amplitude spectrum, and ld is the local
//!    distance.

use std::f32::consts::PI;
use std::sync::atomic::Ordering;

use atomic_float::AtomicF32;
use image::Rgba;
use image::RgbaImage;
use kiddo::SquaredEuclidean;
use kiddo::float::kdtree::KdTree;
use libblur::AnisotropicRadius;
use libblur::BlurImageMut;
use libblur::FastBlurChannels;
use libblur::ThreadingPolicy;
use libblur::stack_blur_f32;
use palette::IntoColor;
use palette::Lab;
use palette::Srgb;
use palette::color_difference::EuclideanDistance;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;
use rayon::slice::ParallelSliceMut;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

use crate::BitPaletteBuilder;
use crate::PaletteBuilder;
use crate::kmeans::parallel_kmeans;
use crate::private;
use crate::rgba_to_lab;

fn par_min_max(data: &[f32]) -> (f32, f32) {
    data.par_iter()
        .copied()
        .fold(
            || (f32::MAX, f32::MIN),
            |(mn, mx), v| (mn.min(v), mx.max(v)),
        )
        .reduce(
            || (f32::MAX, f32::MIN),
            |(a, b), (c, d)| (a.min(c), b.max(d)),
        )
}

fn normalize(data: &mut [f32]) {
    let (min, max) = par_min_max(data);
    let range = (max - min).max(f32::EPSILON);
    data.par_iter_mut().for_each(|v| {
        *v = (*v - min) / range;
    });
}

fn fft_2d(
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

pub struct FocalPaletteBuilder;

impl private::Sealed for FocalPaletteBuilder {}

impl PaletteBuilder for FocalPaletteBuilder {
    const NAME: &'static str = "Focal";

    fn build_palette(
        image: &RgbaImage,
        palette_size: usize,
    ) -> Vec<Lab> {
        let image_width = image.width();
        let image_height = image.height();
        let image = image.to_vec();

        let mut pixels =
            vec![<Lab>::new(0.0, 0.0, 0.0); image_width as usize * image_height as usize];

        image.par_chunks(4).zip(&mut pixels).for_each(|(p, dest)| {
            *dest = rgba_to_lab(Rgba([p[0], p[1], p[2], p[3]]));
        });

        let window_radius = (pixels.len() as f32).ln().clamp(2.0, 16.0) as u32 / 2;

        let mut planner = FftPlanner::new();

        let mut l_values = vec![0.0f32; pixels.len()];
        let mut a_values = vec![0.0f32; pixels.len()];
        let mut b_values = vec![0.0f32; pixels.len()];
        pixels
            .par_iter()
            .copied()
            .zip(&mut l_values)
            .zip(&mut a_values)
            .zip(&mut b_values)
            .for_each(|(((lab, dest_l), dest_a), dest_b)| {
                *dest_l = lab.l;
                *dest_a = lab.a;
                *dest_b = lab.b;
            });

        let l_saliency = compute_saliency(
            &mut planner,
            &l_values,
            image_width,
            image_height,
            window_radius >> 1,
        );

        let a_saliency = compute_saliency(
            &mut planner,
            &a_values,
            image_width,
            image_height,
            window_radius >> 1,
        );

        let b_saliency = compute_saliency(
            &mut planner,
            &b_values,
            image_width,
            image_height,
            window_radius >> 1,
        );

        #[cfg(feature = "dump-l-saliency")]
        {
            let mut quant_l_saliency = vec![0; pixels.len()];

            l_saliency
                .spectral_residual
                .par_iter()
                .copied()
                .zip(&mut quant_l_saliency)
                .for_each(|(d, dest)| {
                    *dest = (d * u8::MAX as f32).round() as u8;
                });
            dump_intermediate("l_sr_aliency", &quant_l_saliency, image_width, image_height);

            l_saliency
                .mod_spectral_residual
                .par_iter()
                .copied()
                .zip(&mut quant_l_saliency)
                .for_each(|(d, dest)| {
                    *dest = (d * u8::MAX as f32).round() as u8;
                });
            dump_intermediate(
                "l_mod_sr_saliency",
                &quant_l_saliency,
                image_width,
                image_height,
            );

            l_saliency
                .phase_spectrum
                .par_iter()
                .copied()
                .zip(&mut quant_l_saliency)
                .for_each(|(d, dest)| {
                    *dest = (d * u8::MAX as f32).round() as u8;
                });
            dump_intermediate(
                "l_phase_saliency",
                &quant_l_saliency,
                image_width,
                image_height,
            );

            l_saliency
                .amplitude_spectrum
                .par_iter()
                .copied()
                .zip(&mut quant_l_saliency)
                .for_each(|(d, dest)| {
                    *dest = (d * u8::MAX as f32).round() as u8;
                });
            dump_intermediate(
                "l_amplitude_saliency",
                &quant_l_saliency,
                image_width,
                image_height,
            );
        }

        #[cfg(feature = "dump-a-saliency")]
        {
            let mut quant_a_saliency = vec![0; pixels.len()];

            a_saliency
                .spectral_residual
                .par_iter()
                .copied()
                .zip(&mut quant_a_saliency)
                .for_each(|(d, dest)| {
                    *dest = (d * u8::MAX as f32).round() as u8;
                });
            dump_intermediate(
                "a_sr_saliency",
                &quant_a_saliency,
                image_width,
                image_height,
            );

            a_saliency
                .mod_spectral_residual
                .par_iter()
                .copied()
                .zip(&mut quant_a_saliency)
                .for_each(|(d, dest)| {
                    *dest = (d * u8::MAX as f32).round() as u8;
                });
            dump_intermediate(
                "a_mod_sr_saliency",
                &quant_a_saliency,
                image_width,
                image_height,
            );

            a_saliency
                .phase_spectrum
                .par_iter()
                .copied()
                .zip(&mut quant_a_saliency)
                .for_each(|(d, dest)| {
                    *dest = (d * u8::MAX as f32).round() as u8;
                });
            dump_intermediate(
                "a_phase_saliency",
                &quant_a_saliency,
                image_width,
                image_height,
            );

            a_saliency
                .amplitude_spectrum
                .par_iter()
                .copied()
                .zip(&mut quant_a_saliency)
                .for_each(|(d, dest)| {
                    *dest = (d * u8::MAX as f32).round() as u8;
                });
            dump_intermediate(
                "a_amplitude_saliency",
                &quant_a_saliency,
                image_width,
                image_height,
            );
        }

        #[cfg(feature = "dump-b-saliency")]
        {
            let mut quant_b_saliency = vec![0; pixels.len()];

            b_saliency
                .spectral_residual
                .par_iter()
                .copied()
                .zip(&mut quant_b_saliency)
                .for_each(|(d, dest)| {
                    *dest = (d * u8::MAX as f32).round() as u8;
                });
            dump_intermediate(
                "b_sr_saliency",
                &quant_b_saliency,
                image_width,
                image_height,
            );

            b_saliency
                .mod_spectral_residual
                .par_iter()
                .copied()
                .zip(&mut quant_b_saliency)
                .for_each(|(d, dest)| {
                    *dest = (d * u8::MAX as f32).round() as u8;
                });
            dump_intermediate(
                "b_mod_sr_saliency",
                &quant_b_saliency,
                image_width,
                image_height,
            );

            b_saliency
                .phase_spectrum
                .par_iter()
                .copied()
                .zip(&mut quant_b_saliency)
                .for_each(|(d, dest)| {
                    *dest = (d * u8::MAX as f32).round() as u8;
                });
            dump_intermediate(
                "b_phase_saliency",
                &quant_b_saliency,
                image_width,
                image_height,
            );

            b_saliency
                .amplitude_spectrum
                .par_iter()
                .copied()
                .zip(&mut quant_b_saliency)
                .for_each(|(d, dest)| {
                    *dest = (d * u8::MAX as f32).round() as u8;
                });
            dump_intermediate(
                "b_amplitude_saliency",
                &quant_b_saliency,
                image_width,
                image_height,
            );
        }

        let centroid = pixels
            .par_iter()
            .copied()
            .reduce(|| <Lab>::new(0.0, 0.0, 0.0), |a, b| a + b)
            / pixels.len() as f32;

        let mut dists = vec![0.0f32; pixels.len()];
        pixels
            .par_iter()
            .copied()
            .zip(&mut dists)
            .for_each(|(p, dest)| {
                *dest = centroid.distance(p);
            });

        normalize(&mut dists);

        let mut local_dists = vec![0.0f32; pixels.len()];

        (0..pixels.len())
            .into_par_iter()
            .zip(&mut local_dists)
            .for_each(|(idx, dest)| {
                let x = idx as isize % image_width as isize;
                let y = idx as isize / image_width as isize;

                let mut sum = 0.0;
                let mut count = 0;
                for dx in -(window_radius as isize)..=window_radius as isize {
                    for dy in -(window_radius as isize)..=window_radius as isize {
                        let x = x + dx;
                        let y = y + dy;

                        if (dx == 0 && dy == 0)
                            || x < 0
                            || y < 0
                            || x >= image_width as isize
                            || y >= image_height as isize
                        {
                            continue;
                        }
                        let jdx = y * image_width as isize + x;
                        sum += pixels[jdx as usize].distance(pixels[idx]);
                        count += 1;
                    }
                }

                *dest = sum / count as f32
            });

        normalize(&mut local_dists);

        #[cfg(feature = "dump-local-saliency")]
        {
            let mut quant_local_dists = vec![0; pixels.len()];
            local_dists
                .par_iter()
                .copied()
                .zip(&mut quant_local_dists)
                .for_each(|(d, dest)| {
                    *dest = (d * u8::MAX as f32).round() as u8;
                });

            dump_intermediate("local_dists", &quant_local_dists, image_width, image_height);
        }

        let mut candidates = vec![(<Lab>::new(0.0, 0.0, 0.0), 0.0); pixels.len()];
        pixels
            .into_par_iter()
            .zip(l_saliency.spectral_residual)
            .zip(l_saliency.mod_spectral_residual)
            .zip(l_saliency.phase_spectrum)
            .zip(l_saliency.amplitude_spectrum)
            .zip(a_saliency.spectral_residual)
            .zip(a_saliency.mod_spectral_residual)
            .zip(a_saliency.phase_spectrum)
            .zip(a_saliency.amplitude_spectrum)
            .zip(b_saliency.spectral_residual)
            .zip(b_saliency.mod_spectral_residual)
            .zip(b_saliency.phase_spectrum)
            .zip(b_saliency.amplitude_spectrum)
            .zip(local_dists)
            .zip(dists)
            .zip(&mut candidates)
            .for_each(
                |(
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (((((lab, l_sr), l_msr), l_p), l_a), a_sr),
                                                        a_msr,
                                                    ),
                                                    a_p,
                                                ),
                                                a_a,
                                            ),
                                            b_sr,
                                        ),
                                        b_msr,
                                    ),
                                    b_p,
                                ),
                                b_a,
                            ),
                            local,
                        ),
                        global,
                    ),
                    dest,
                )| {
                    let outliers = (0.5 * l_sr + 0.25 * a_sr + 0.25 * b_sr)
                        .max(0.5 * l_msr + 0.25 * a_msr + 0.25 * b_msr);
                    let edges = l_p * 0.5 + a_p * 0.25 + b_p * 0.25;
                    let blobs = l_a * 0.5 + a_a * 0.25 + b_a * 0.25;

                    let modifier = global.max(local);
                    let w = (outliers * 0.7 + edges * 0.2 + blobs * 0.1) * modifier;

                    *dest = (lab, w);
                },
            );

        {
            let (min_weight, max_weight) = candidates
                .par_iter()
                .map(|(_, w)| *w)
                .fold(
                    || (f32::MAX, f32::MIN),
                    |(mn, mx), v| (mn.min(v), mx.max(v)),
                )
                .reduce(
                    || (f32::MAX, f32::MIN),
                    |(a, b), (c, d)| (a.min(c), b.max(d)),
                );
            let range = (max_weight - min_weight).max(f32::EPSILON);
            candidates.par_iter_mut().for_each(|(_, w)| {
                *w = (*w - min_weight) / range;
            });
        }

        #[cfg(feature = "dump-weights")]
        {
            let mut quant_candidates = vec![0; candidates.len()];
            candidates
                .par_iter()
                .copied()
                .zip(&mut quant_candidates)
                .for_each(|((_, w), dest)| {
                    *dest = (w * u8::MAX as f32).round() as u8;
                });

            dump_intermediate("candidates", &quant_candidates, image_width, image_height);
        }

        o_means(candidates, palette_size)
    }
}

struct Saliency {
    spectral_residual: Vec<f32>,
    mod_spectral_residual: Vec<f32>,
    phase_spectrum: Vec<f32>,
    amplitude_spectrum: Vec<f32>,
}

fn blur_norm_sqr_normalize(
    buffer: &[Complex<f32>],
    width_p2: u32,
    height_p2: u32,
    window_radius: u32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; buffer.len()];
    buffer
        .par_iter()
        .copied()
        .zip(&mut out)
        .for_each(|(c, dest)| *dest = c.norm_sqr());

    {
        let mut img = BlurImageMut::borrow(&mut out, width_p2, height_p2, FastBlurChannels::Plane);
        stack_blur_f32(
            &mut img,
            AnisotropicRadius::new(window_radius),
            ThreadingPolicy::Adaptive,
        )
        .unwrap();
    }

    normalize(&mut out);
    out
}

fn crop_p2(
    data: Vec<f32>,
    w: u32,
    h: u32,
    wp2: u32,
    hp2: u32,
) -> Vec<f32> {
    if wp2 == w && hp2 == h {
        return data;
    }
    let mut out = vec![0.0f32; h as usize * w as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            out[y * w as usize + x] = data[y * wp2 as usize + x];
        }
    }
    out
}

fn spectral_residual(
    planner: &mut FftPlanner<f32>,
    scratch: &mut Vec<Complex<f32>>,
    amplitude: &[f32],
    phase: &[f32],
    width_p2: usize,
    height_p2: usize,
    window_radius: u32,
) -> Vec<f32> {
    let mut log_amplitude = vec![0.0f32; phase.len()];
    amplitude
        .par_iter()
        .copied()
        .zip(&mut log_amplitude)
        .for_each(|(a, dest)| *dest = a.max(f32::EPSILON).ln());

    let mut log_blurred = log_amplitude.clone();

    {
        let mut img = BlurImageMut::borrow(
            &mut log_blurred,
            width_p2 as u32,
            height_p2 as u32,
            FastBlurChannels::Plane,
        );
        stack_blur_f32(
            &mut img,
            AnisotropicRadius::new(window_radius),
            ThreadingPolicy::Adaptive,
        )
        .unwrap();
    }

    let mut residual = vec![0.0f32; width_p2 * height_p2];
    log_amplitude
        .into_par_iter()
        .zip(log_blurred)
        .zip(&mut residual)
        .for_each(|((a, b), dest)| {
            *dest = a - b;
        });

    let ifft_row = planner.plan_fft_inverse(width_p2);
    let ifft_col = planner.plan_fft_inverse(height_p2);

    let mut buffer = vec![Complex::zero(); residual.len()];
    residual
        .into_par_iter()
        .zip(phase.par_iter())
        .zip(&mut buffer)
        .for_each(|((a, p), dest)| {
            *dest = Complex {
                re: a.exp() * p.cos(),
                im: a.exp() * p.sin(),
            };
        });

    fft_2d(
        &mut buffer,
        scratch,
        width_p2,
        height_p2,
        ifft_row.as_ref(),
        ifft_col.as_ref(),
    );

    blur_norm_sqr_normalize(&buffer, width_p2 as u32, height_p2 as u32, window_radius)
}

fn flatten_midrange(
    values: &[f32],
    min_val: f32,
    max_val: f32,
) -> Vec<f32> {
    let range = (max_val - min_val).max(f32::EPSILON);
    let mut out = vec![0.0f32; values.len()];
    values
        .par_iter()
        .copied()
        .zip(&mut out)
        .for_each(|(v, dest)| {
            let x = ((v - min_val) / range - 0.5) * 2.0;
            *dest = ((x * PI / 2.0).sin().powi(15) / 2.0 + 0.5) * (max_val - min_val) + min_val;
        });
    out
}

fn compute_saliency(
    planner: &mut FftPlanner<f32>,
    channel_values: &[f32],
    image_width: u32,
    image_height: u32,
    window_radius: u32,
) -> Saliency {
    let wp2 = image_width.next_power_of_two();
    let hp2 = image_height.next_power_of_two();
    let w = wp2 as usize;
    let h = hp2 as usize;

    let fft_row = planner.plan_fft_forward(w);
    let fft_col = planner.plan_fft_forward(h);

    let mut buffer = vec![Complex::zero(); w * h];
    channel_values
        .par_iter()
        .copied()
        .zip(&mut buffer)
        .for_each(|(v, dest)| {
            *dest = Complex { re: v, im: 0.0 };
        });

    let mut scratch = Vec::new();
    fft_2d(
        &mut buffer,
        &mut scratch,
        w,
        h,
        fft_row.as_ref(),
        fft_col.as_ref(),
    );

    let mut amplitude = vec![0.0f32; buffer.len()];
    buffer
        .par_iter()
        .copied()
        .zip(&mut amplitude)
        .for_each(|(c, dest)| *dest = c.norm());

    // Phase spectrum (PFT): IFFT of unit-phase vectors
    let pft = {
        let mut ifft_buffer = vec![Complex::zero(); buffer.len()];

        amplitude
            .par_iter()
            .copied()
            .zip(&buffer)
            .zip(&mut ifft_buffer)
            .for_each(|((a, c), dest)| {
                *dest = Complex {
                    re: if a > f32::EPSILON { c.re / a } else { 0.0 },
                    im: if a > f32::EPSILON { c.im / a } else { 0.0 },
                };
            });

        let ifft_row = planner.plan_fft_inverse(w);
        let ifft_col = planner.plan_fft_inverse(h);
        fft_2d(
            &mut ifft_buffer,
            &mut scratch,
            w,
            h,
            ifft_row.as_ref(),
            ifft_col.as_ref(),
        );

        blur_norm_sqr_normalize(&ifft_buffer, wp2, hp2, window_radius)
    };

    let mut phase = vec![0.0f32; buffer.len()];
    buffer
        .into_par_iter()
        .zip(&mut phase)
        .for_each(|(c, dest)| *dest = c.arg());

    let (min_amplitude, max_amplitude) = par_min_max(&amplitude);
    let mod_amplitude = flatten_midrange(&amplitude, min_amplitude, max_amplitude);

    // Amplitude spectrum (ASR): IFFT with modulated amplitude + original phase
    let asr = {
        let mut ifft_buffer = vec![Complex::zero(); phase.len()];
        mod_amplitude
            .par_iter()
            .copied()
            .zip(&phase)
            .zip(&mut ifft_buffer)
            .for_each(|((a, p), dest)| {
                *dest = Complex {
                    re: a * p.cos(),
                    im: a * p.sin(),
                };
            });

        let ifft_row = planner.plan_fft_inverse(w);
        let ifft_col = planner.plan_fft_inverse(h);
        fft_2d(
            &mut ifft_buffer,
            &mut scratch,
            w,
            h,
            ifft_row.as_ref(),
            ifft_col.as_ref(),
        );

        blur_norm_sqr_normalize(&ifft_buffer, wp2, hp2, window_radius)
    };

    // Spectral residual (SR): log-amplitude residual + original phase
    let sr = spectral_residual(
        planner,
        &mut scratch,
        &amplitude,
        &phase,
        w,
        h,
        window_radius,
    );

    // Modulated spectral residual (MSR): log-modulated-amplitude residual +
    // modulated phase
    let msr = {
        let (min_phase, max_phase) = par_min_max(&phase);
        let mod_phase = flatten_midrange(&phase, min_phase, max_phase);

        spectral_residual(
            planner,
            &mut scratch,
            &mod_amplitude,
            &mod_phase,
            w,
            h,
            window_radius,
        )
    };

    Saliency {
        spectral_residual: crop_p2(sr, image_width, image_height, wp2, hp2),
        mod_spectral_residual: crop_p2(msr, image_width, image_height, wp2, hp2),
        phase_spectrum: crop_p2(pft, image_width, image_height, wp2, hp2),
        amplitude_spectrum: crop_p2(asr, image_width, image_height, wp2, hp2),
    }
}

#[cfg(any(
    feature = "dump-l-saliency",
    feature = "dump-a-saliency",
    feature = "dump-b-saliency",
    feature = "dump-local-saliency",
    feature = "dump-weights"
))]
fn dump_intermediate(
    name: &str,
    buffer: &[u8],
    width: u32,
    height: u32,
) {
    use std::hash::BuildHasher;
    use std::hash::Hasher;
    use std::hash::RandomState;
    let rand = BuildHasher::build_hasher(&RandomState::new()).finish();
    image::save_buffer(
        format!("{name}-{rand}.png"),
        buffer,
        width,
        height,
        image::ColorType::L8,
    )
    .unwrap()
}

const MAX_OUTER_ITERATIONS: usize = 32;

pub(crate) fn o_means(
    mut candidates: Vec<(Lab, f32)>,
    palette_size: usize,
) -> Vec<Lab> {
    let mut final_clusters = vec![];

    for _ in 0..MAX_OUTER_ITERATIONS {
        let (color, weight) = candidates
            .par_iter()
            .copied()
            .map(|(c, w)| (c * w, w))
            .reduce(
                || (<Lab>::new(0.0, 0.0, 0.0), 0.0),
                |a, b| (a.0 + b.0, a.1 + b.1),
            );

        let centroid = color / weight.max(f32::EPSILON);

        let var_dist = candidates
            .par_iter()
            .map(|(c, w)| c.distance_squared(centroid) * w)
            .sum::<f32>()
            / weight.max(f32::EPSILON);

        let sigma = var_dist.sqrt() * 2.0;

        let summaries = std::iter::repeat_with(|| {
            (
                [
                    AtomicF32::new(0.0),
                    AtomicF32::new(0.0),
                    AtomicF32::new(0.0),
                ],
                AtomicF32::new(0.0),
            )
        })
        .take(palette_size)
        .collect::<Vec<_>>();

        let shift = BitPaletteBuilder::shift(palette_size);
        candidates.par_iter().copied().for_each(|(color, weight)| {
            let rgb: Srgb = color.into_color();
            let slot = BitPaletteBuilder::index(rgb.into_format(), shift);
            summaries[slot].0[0].fetch_add(rgb.red * weight, Ordering::Relaxed);
            summaries[slot].0[1].fetch_add(rgb.green * weight, Ordering::Relaxed);
            summaries[slot].0[2].fetch_add(rgb.blue * weight, Ordering::Relaxed);
            summaries[slot].1.fetch_add(weight, Ordering::Relaxed);
        });

        let mut centroids = KdTree::<_, _, 3, 257, u32>::with_capacity(256);

        for (idx, (color, weight)) in summaries.iter().enumerate() {
            let count = weight.load(Ordering::Relaxed);
            if count > 0.0 {
                let color = Srgb::new(
                    color[0].load(Ordering::Relaxed) / count,
                    color[1].load(Ordering::Relaxed) / count,
                    color[2].load(Ordering::Relaxed) / count,
                );
                let color: Lab = color.into_color();

                centroids.add(&[color.l, color.a, color.b], idx as u32);
            }
        }

        let mut cluster_assignments = vec![-1i64; candidates.len()];
        candidates
            .par_iter()
            .copied()
            .zip(&mut cluster_assignments)
            .for_each(|((color, _), slot)| {
                let nearest =
                    centroids.nearest_one::<SquaredEuclidean>(&[color.l, color.a, color.b]);

                if nearest.distance <= sigma {
                    *slot = nearest.item as i64;
                }
            });

        // Accumulate initial cluster means from assignments
        let mut cluster_sums: Vec<[f32; 3]> = vec![[0.0; 3]; palette_size];
        let mut cluster_weights: Vec<f32> = vec![0.0; palette_size];

        for (idx, &slot) in cluster_assignments.iter().enumerate() {
            if slot >= 0 {
                let s = slot as usize;
                let w = candidates[idx].1;
                cluster_sums[s][0] += candidates[idx].0.l * w;
                cluster_sums[s][1] += candidates[idx].0.a * w;
                cluster_sums[s][2] += candidates[idx].0.b * w;
                cluster_weights[s] += w;
            }
        }

        for _ in 0..100 {
            // Snapshot centroids from current means
            centroids = KdTree::<_, _, 3, 257, u32>::with_capacity(256);

            for (cidx, (sum, &count)) in cluster_sums.iter().zip(&cluster_weights).enumerate() {
                if count > 0.0 {
                    centroids.add(
                        &[sum[0] / count, sum[1] / count, sum[2] / count],
                        cidx as u32,
                    );
                }
            }

            // Compute new assignments (read-only phase)
            let new_assignments: Vec<i64> = candidates
                .par_iter()
                .copied()
                .map(|(color, _)| {
                    let nearest =
                        centroids.nearest_one::<SquaredEuclidean>(&[color.l, color.a, color.b]);

                    if nearest.distance > sigma {
                        -1
                    } else {
                        nearest.item as i64
                    }
                })
                .collect();

            // Apply deltas sequentially from assignment changes
            let mut shifts = 0usize;
            for (idx, (&new_slot, old_slot)) in new_assignments
                .iter()
                .zip(&mut cluster_assignments)
                .enumerate()
            {
                if new_slot == *old_slot {
                    continue;
                }
                shifts += 1;

                let (color, w) = candidates[idx];
                let wl = color.l * w;
                let wa = color.a * w;
                let wb = color.b * w;

                if *old_slot >= 0 {
                    let s = *old_slot as usize;
                    cluster_sums[s][0] -= wl;
                    cluster_sums[s][1] -= wa;
                    cluster_sums[s][2] -= wb;
                    cluster_weights[s] -= w;
                }

                if new_slot >= 0 {
                    let s = new_slot as usize;
                    cluster_sums[s][0] += wl;
                    cluster_sums[s][1] += wa;
                    cluster_sums[s][2] += wb;
                    cluster_weights[s] += w;
                }

                *old_slot = new_slot;
            }

            if shifts == 0 {
                break;
            }
        }

        for (sum, &weight) in cluster_sums.iter().zip(&cluster_weights) {
            if weight > 0.0 {
                final_clusters.push((
                    Lab::new(sum[0] / weight, sum[1] / weight, sum[2] / weight),
                    weight,
                ));
            }
        }

        let outliers = cluster_assignments
            .iter()
            .enumerate()
            .filter(|(_, cluster)| **cluster < 0)
            .map(|(i, _)| candidates[i])
            .collect::<Vec<_>>();

        if outliers.is_empty() || outliers.len() == candidates.len() {
            final_clusters.extend(outliers);
            break;
        }
        candidates = outliers;
    }

    if final_clusters.len() <= palette_size {
        final_clusters.into_iter().map(|(c, _)| c).collect()
    } else {
        parallel_kmeans(&final_clusters, palette_size).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_square_non_p2_image() {
        let mut planner = FftPlanner::<f32>::new();
        let channel_values = vec![0.5; 122 * 64];
        let image_width = 122;
        let image_height = 64;
        let window_radius = 3;

        let saliency: Saliency = compute_saliency(
            &mut planner,
            &channel_values,
            image_width,
            image_height,
            window_radius,
        );

        assert_eq!(saliency.spectral_residual.len(), channel_values.len());
        assert_eq!(saliency.mod_spectral_residual.len(), channel_values.len());
        assert_eq!(saliency.phase_spectrum.len(), channel_values.len());
        assert_eq!(saliency.amplitude_spectrum.len(), channel_values.len());
    }
}
