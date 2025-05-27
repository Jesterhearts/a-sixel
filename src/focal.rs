//! Use weighted pixels based on the image's spectral properties to provided
//! weighted input to kmeans.
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

use image::{
    Rgb,
    RgbImage,
};
use libblur::{
    stack_blur_f32,
    AnisotropicRadius,
    BlurImageMut,
    FastBlurChannels,
    ThreadingPolicy,
};
use palette::{
    color_difference::EuclideanDistance,
    Lab,
};
use rayon::{
    iter::{
        IndexedParallelIterator,
        IntoParallelIterator,
        IntoParallelRefIterator,
        IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::{
        ParallelSlice,
        ParallelSliceMut,
    },
};
use rustfft::{
    num_complex::Complex,
    num_traits::Zero,
    FftPlanner,
};

use crate::{
    dither::Sierra,
    kmeans::parallel_kmeans,
    private,
    rgb_to_lab,
    PaletteBuilder,
    SixelEncoder,
};

pub type FocalSixelEncoderMono<D = Sierra> = SixelEncoder<FocalPaletteBuilder<2>, D>;
pub type FocalSixelEncoder4<D = Sierra> = SixelEncoder<FocalPaletteBuilder<4>, D>;
pub type FocalSixelEncoder8<D = Sierra> = SixelEncoder<FocalPaletteBuilder<8>, D>;
pub type FocalSixelEncoder16<D = Sierra> = SixelEncoder<FocalPaletteBuilder<16>, D>;
pub type FocalSixelEncoder32<D = Sierra> = SixelEncoder<FocalPaletteBuilder<32>, D>;
pub type FocalSixelEncoder64<D = Sierra> = SixelEncoder<FocalPaletteBuilder<64>, D>;
pub type FocalSixelEncoder128<D = Sierra> = SixelEncoder<FocalPaletteBuilder<128>, D>;
pub type FocalSixelEncoder256<D = Sierra> = SixelEncoder<FocalPaletteBuilder<256>, D>;

pub struct FocalPaletteBuilder<const PALETTE_SIZE: usize = 256>;

impl<const PALETTE_SIZE: usize> private::Sealed for FocalPaletteBuilder<PALETTE_SIZE> {}

impl<const PALETTE_SIZE: usize> PaletteBuilder for FocalPaletteBuilder<PALETTE_SIZE> {
    const PALETTE_SIZE: usize = PALETTE_SIZE;

    #[inline(never)]
    fn build_palette(image: &RgbImage) -> Vec<Lab> {
        let image_width = image.width();
        let image_height = image.height();
        let image = image.to_vec();

        let mut pixels =
            vec![<Lab>::new(0.0, 0.0, 0.0); image_width as usize * image_height as usize];

        image.par_chunks(3).zip(&mut pixels).for_each(|(p, dest)| {
            *dest = rgb_to_lab(Rgb([p[0], p[1], p[2]]));
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

        #[cfg(feature = "dump_l_saliency")]
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

        #[cfg(feature = "dump_a_saliency")]
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

        #[cfg(feature = "dump_b_saliency")]
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

        let max_dist = (<Lab>::max_l() - <Lab>::min_l()).powi(2)
            + (<Lab>::max_a() - <Lab>::min_a()).powi(2)
            + (<Lab>::max_b() - <Lab>::min_b()).powi(2);
        let mut local_dists = vec![0.0f32; pixels.len()];

        (0..pixels.len())
            .into_par_iter()
            .zip(&mut local_dists)
            .for_each(|(idx, dest)| {
                let x = idx as isize % image_width as isize;
                let y = idx as isize / image_height as isize;

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

                *dest = (sum / count as f32) / max_dist
            });

        #[cfg(feature = "dump_local_saliency")]
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
                                                ((((((lab, l_sr), l_msr), l_p), l_a), a_sr), a_msr),
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
                    dest,
                )| {
                    let outlier_outlier = l_msr.max(a_msr).max(b_msr);
                    let outliers = l_sr.max(a_sr).max(b_sr).max(outlier_outlier);
                    let edges = l_p.max(a_p).max(b_p);
                    let blobs = l_a.max(a_a).max(b_a);

                    // Lerp between blobs <=> outliers <=> edges using local distance. Local
                    // distance is *high* for edges and features and *low* for blobs/planes of
                    // colors. If it's neither, then it's probably important if it's an outlier.
                    let w = if local <= 0.5 {
                        blobs + 2.0 * local * (outliers - blobs)
                    } else {
                        outliers + 2.0 * (local - 0.5) * (edges - outliers)
                    };

                    *dest = (lab, w);
                },
            );

        #[cfg(feature = "dump_weights")]
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

        parallel_kmeans::<PALETTE_SIZE>(&candidates)
    }
}

struct Saliency {
    spectral_residual: Vec<f32>,
    mod_spectral_residual: Vec<f32>,
    phase_spectrum: Vec<f32>,
    amplitude_spectrum: Vec<f32>,
}

fn compute_saliency(
    planner: &mut FftPlanner<f32>,
    channel_values: &[f32],
    image_width: u32,
    image_height: u32,
    window_radius: u32,
) -> Saliency {
    let buffer = {
        let fft_row = planner.plan_fft_forward(image_width as usize);
        let fft_col = planner.plan_fft_forward(image_height as usize);

        let mut buffer = vec![Complex::zero(); channel_values.len()];
        channel_values
            .par_iter()
            .copied()
            .zip(&mut buffer)
            .for_each(|(v, dest)| {
                *dest = Complex { re: v, im: 0.0 };
            });

        buffer
            .par_chunks_mut(image_width as usize)
            .for_each(|chunk| {
                fft_row.process(chunk);
            });

        let mut transpose = vec![Complex::zero(); buffer.len()];
        (0..image_width as usize)
            .into_par_iter()
            .zip(transpose.par_chunks_mut(image_height as usize))
            .for_each(|(x, col)| {
                (0..image_height as usize)
                    .into_par_iter()
                    .zip(col)
                    .for_each(|(y, dest)| {
                        *dest = buffer[y * image_width as usize + x];
                    });
            });

        transpose
            .par_chunks_mut(image_height as usize)
            .for_each(|chunk| {
                fft_col.process(chunk);
            });

        (0..image_height as usize)
            .into_par_iter()
            .zip(buffer.par_chunks_mut(image_width as usize))
            .for_each(|(y, row)| {
                (0..image_width as usize)
                    .into_par_iter()
                    .zip(row)
                    .for_each(|(x, dest)| {
                        *dest = transpose[x * image_height as usize + y];
                    });
            });

        transpose
    };

    let mut amplitude = vec![0.0f32; buffer.len()];
    buffer
        .par_iter()
        .copied()
        .zip(&mut amplitude)
        .for_each(|(c, dest)| *dest = c.norm());

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

        let ifft_row = planner.plan_fft_inverse(image_width as usize);
        let ifft_col = planner.plan_fft_inverse(image_height as usize);

        ifft_buffer
            .par_chunks_mut(image_height as usize)
            .for_each(|chunk| {
                ifft_col.process(chunk);
            });

        let mut transpose = vec![Complex::zero(); ifft_buffer.len()];
        (0..image_width as usize)
            .into_par_iter()
            .zip(transpose.par_chunks_mut(image_height as usize))
            .for_each(|(x, col)| {
                (0..image_height as usize)
                    .into_par_iter()
                    .zip(col)
                    .for_each(|(y, dest)| {
                        *dest = ifft_buffer[y * image_width as usize + x];
                    });
            });

        transpose
            .par_chunks_mut(image_height as usize)
            .for_each(|chunk| {
                ifft_row.process(chunk);
            });

        (0..image_height as usize)
            .into_par_iter()
            .zip(ifft_buffer.par_chunks_mut(image_width as usize))
            .for_each(|(y, row)| {
                (0..image_width as usize)
                    .into_par_iter()
                    .zip(row)
                    .for_each(|(x, dest)| {
                        *dest = transpose[x * image_height as usize + y];
                    });
            });

        let mut pft = vec![0.0f32; transpose.len()];

        transpose
            .par_iter()
            .copied()
            .zip(&mut pft)
            .for_each(|(c, dest)| *dest = c.norm_sqr());

        {
            let mut pft =
                BlurImageMut::borrow(&mut pft, image_width, image_height, FastBlurChannels::Plane);
            stack_blur_f32(
                &mut pft,
                AnisotropicRadius::new(window_radius),
                ThreadingPolicy::Adaptive,
            )
            .unwrap();
        }

        let max_pft = pft.par_iter().copied().reduce(|| f32::MIN, f32::max);
        let min_pft = pft.par_iter().copied().reduce(|| f32::MAX, f32::min);
        pft.par_iter_mut().for_each(|a| {
            *a = (*a - min_pft) / (max_pft - min_pft).max(f32::EPSILON);
        });

        pft
    };

    let mut phase = vec![0.0f32; buffer.len()];
    buffer
        .into_par_iter()
        .zip(&mut phase)
        .for_each(|(c, dest)| *dest = c.arg());

    let mut mod_amplitude = vec![0.0f32; phase.len()];
    let max_amplitude = amplitude.par_iter().copied().reduce(|| f32::MIN, f32::max);
    let min_amplitude = amplitude.par_iter().copied().reduce(|| f32::MAX, f32::min);
    amplitude
        .par_iter()
        .copied()
        .zip(&mut mod_amplitude)
        .for_each(|(a, dest)| {
            let x = ((a - min_amplitude) / (max_amplitude - min_amplitude).max(f32::EPSILON) - 0.5)
                * 2.0;
            *dest = ((x * PI / 2.0).sin().powi(15) / 2.0 + 0.5) * (max_amplitude - min_amplitude)
                + min_amplitude
        });

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

        let ifft_row = planner.plan_fft_inverse(image_width as usize);
        let ifft_col = planner.plan_fft_inverse(image_height as usize);
        ifft_buffer
            .par_chunks_mut(image_height as usize)
            .for_each(|chunk| {
                ifft_col.process(chunk);
            });

        let mut transpose = vec![Complex::zero(); ifft_buffer.len()];
        (0..image_width as usize)
            .into_par_iter()
            .zip(transpose.par_chunks_mut(image_height as usize))
            .for_each(|(x, col)| {
                (0..image_height as usize)
                    .into_par_iter()
                    .zip(col)
                    .for_each(|(y, dest)| {
                        *dest = ifft_buffer[y * image_width as usize + x];
                    });
            });

        transpose
            .par_chunks_mut(image_height as usize)
            .for_each(|chunk| {
                ifft_row.process(chunk);
            });

        (0..image_height as usize)
            .into_par_iter()
            .zip(ifft_buffer.par_chunks_mut(image_width as usize))
            .for_each(|(y, row)| {
                (0..image_width as usize)
                    .into_par_iter()
                    .zip(row)
                    .for_each(|(x, dest)| {
                        *dest = transpose[x * image_height as usize + y];
                    });
            });

        let mut asr = vec![0.0f32; transpose.len()];
        transpose
            .par_iter()
            .copied()
            .zip(&mut asr)
            .for_each(|(c, dest)| *dest = c.norm_sqr());

        {
            let mut asr =
                BlurImageMut::borrow(&mut asr, image_width, image_height, FastBlurChannels::Plane);
            stack_blur_f32(
                &mut asr,
                AnisotropicRadius::new(window_radius),
                ThreadingPolicy::Adaptive,
            )
            .unwrap();
        }

        let max_asr = asr.par_iter().copied().reduce(|| f32::MIN, f32::max);
        let min_asr = asr.par_iter().copied().reduce(|| f32::MAX, f32::min);
        asr.par_iter_mut().for_each(|a| {
            *a = (*a - min_asr) / (max_asr - min_asr).max(f32::EPSILON);
        });

        asr
    };

    let sr = {
        let mut log_amplitude = vec![0.0f32; phase.len()];
        amplitude
            .into_par_iter()
            .zip(&mut log_amplitude)
            .for_each(|(a, dest)| *dest = a.max(f32::EPSILON).ln());

        let mut log_blurred = log_amplitude.clone();

        {
            let mut amplitude = BlurImageMut::borrow(
                &mut log_blurred,
                image_width,
                image_height,
                FastBlurChannels::Plane,
            );
            stack_blur_f32(
                &mut amplitude,
                AnisotropicRadius::new(window_radius),
                ThreadingPolicy::Adaptive,
            )
            .unwrap();
        }

        let mut residual = vec![0.0f32; image_width as usize * image_height as usize];
        log_amplitude
            .into_par_iter()
            .zip(log_blurred)
            .zip(&mut residual)
            .for_each(|((a, b), dest)| {
                *dest = a - b;
            });

        let sr_buffer = {
            let ifft_row = planner.plan_fft_inverse(image_width as usize);
            let ifft_col = planner.plan_fft_inverse(image_height as usize);

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

            buffer
                .par_chunks_mut(image_height as usize)
                .for_each(|chunk| {
                    ifft_col.process(chunk);
                });

            let mut transpose = vec![Complex::zero(); buffer.len()];
            (0..image_width as usize)
                .into_par_iter()
                .zip(transpose.par_chunks_mut(image_height as usize))
                .for_each(|(x, col)| {
                    (0..image_height as usize)
                        .into_par_iter()
                        .zip(col)
                        .for_each(|(y, dest)| {
                            *dest = buffer[y * image_width as usize + x];
                        });
                });

            transpose
                .par_chunks_mut(image_height as usize)
                .for_each(|chunk| {
                    ifft_row.process(chunk);
                });

            (0..image_height as usize)
                .into_par_iter()
                .zip(buffer.par_chunks_mut(image_width as usize))
                .for_each(|(y, row)| {
                    (0..image_width as usize)
                        .into_par_iter()
                        .zip(row)
                        .for_each(|(x, dest)| {
                            *dest = transpose[x * image_height as usize + y];
                        });
                });

            transpose
        };

        let mut saliency = vec![0.0f32; image_width as usize * image_height as usize];
        sr_buffer
            .into_par_iter()
            .zip(&mut saliency)
            .for_each(|(c, dest)| *dest = c.norm_sqr());

        {
            let mut saliency = BlurImageMut::borrow(
                &mut saliency,
                image_width,
                image_height,
                FastBlurChannels::Plane,
            );
            stack_blur_f32(
                &mut saliency,
                AnisotropicRadius::new(window_radius),
                ThreadingPolicy::Adaptive,
            )
            .unwrap();
        }

        let max_saliency = saliency.par_iter().copied().reduce(|| f32::MIN, f32::max);
        let min_saliency = saliency.par_iter().copied().reduce(|| f32::MAX, f32::min);
        saliency.par_iter_mut().for_each(|s| {
            *s = (*s - min_saliency) / (max_saliency - min_saliency).max(f32::EPSILON);
        });

        saliency
    };

    let msr = {
        let mut log_amplitude = vec![0.0f32; phase.len()];
        mod_amplitude
            .into_par_iter()
            .zip(&mut log_amplitude)
            .for_each(|(a, dest)| *dest = a.max(f32::EPSILON).ln());

        let mut log_blurred = log_amplitude.clone();

        {
            let mut amplitude = BlurImageMut::borrow(
                &mut log_blurred,
                image_width,
                image_height,
                FastBlurChannels::Plane,
            );
            stack_blur_f32(
                &mut amplitude,
                AnisotropicRadius::new(window_radius),
                ThreadingPolicy::Adaptive,
            )
            .unwrap();
        }

        let mut residual = vec![0.0f32; image_width as usize * image_height as usize];
        log_amplitude
            .into_par_iter()
            .zip(log_blurred)
            .zip(&mut residual)
            .for_each(|((a, b), dest)| {
                *dest = a - b;
            });

        let mut mod_phase = vec![0.0f32; phase.len()];

        let max_phase = phase.par_iter().copied().reduce(|| f32::MIN, f32::max);
        let min_phase = phase.par_iter().copied().reduce(|| f32::MAX, f32::min);

        phase
            .into_par_iter()
            .zip(&mut mod_phase)
            .for_each(|(p, dest)| {
                let x = ((p - min_phase) / (max_phase - min_phase).max(f32::EPSILON) - 0.5) * 2.0;
                *dest = ((x * PI / 2.0).sin().powi(15) / 2.0 + 0.5) * (max_phase - min_phase)
                    + min_phase;
            });

        let mut msr_buffer = vec![Complex::zero(); mod_phase.len()];
        residual
            .into_par_iter()
            .zip(mod_phase)
            .zip(&mut msr_buffer)
            .for_each(|((a, p), dest)| {
                *dest = Complex {
                    re: a * p.cos(),
                    im: a * p.sin(),
                };
            });

        let ifft_row = planner.plan_fft_inverse(image_width as usize);
        let ifft_col = planner.plan_fft_inverse(image_height as usize);
        msr_buffer
            .par_chunks_mut(image_height as usize)
            .for_each(|chunk| {
                ifft_col.process(chunk);
            });

        let mut transpose = vec![Complex::zero(); msr_buffer.len()];
        (0..image_width as usize)
            .into_par_iter()
            .zip(transpose.par_chunks_mut(image_height as usize))
            .for_each(|(x, col)| {
                (0..image_height as usize)
                    .into_par_iter()
                    .zip(col)
                    .for_each(|(y, dest)| {
                        *dest = msr_buffer[y * image_width as usize + x];
                    });
            });

        transpose
            .par_chunks_mut(image_height as usize)
            .for_each(|chunk| {
                ifft_row.process(chunk);
            });

        (0..image_height as usize)
            .into_par_iter()
            .zip(msr_buffer.par_chunks_mut(image_width as usize))
            .for_each(|(y, row)| {
                (0..image_width as usize)
                    .into_par_iter()
                    .zip(row)
                    .for_each(|(x, dest)| {
                        *dest = transpose[x * image_height as usize + y];
                    });
            });

        let mut msr = vec![0.0f32; transpose.len()];
        transpose
            .par_iter()
            .copied()
            .zip(&mut msr)
            .for_each(|(c, dest)| *dest = c.norm_sqr());

        {
            let mut msr =
                BlurImageMut::borrow(&mut msr, image_width, image_height, FastBlurChannels::Plane);
            stack_blur_f32(
                &mut msr,
                AnisotropicRadius::new(window_radius),
                ThreadingPolicy::Adaptive,
            )
            .unwrap();
        }

        let max_msr = msr.par_iter().copied().reduce(|| f32::MIN, f32::max);
        let min_msr = msr.par_iter().copied().reduce(|| f32::MAX, f32::min);
        msr.par_iter_mut().for_each(|s| {
            *s = (*s - min_msr) / (max_msr - min_msr).max(f32::EPSILON);
        });

        msr
    };

    Saliency {
        spectral_residual: sr,
        mod_spectral_residual: msr,
        phase_spectrum: pft,
        amplitude_spectrum: asr,
    }
}

#[cfg(any(
    feature = "dump_l_saliency",
    feature = "dump_a_saliency",
    feature = "dump_b_saliency",
    feature = "dump_dist_saliency",
    feature = "dump_local_saliency",
    feature = "dump_weights"
))]
fn dump_intermediate(name: &str, buffer: &[u8], width: u32, height: u32) {
    use std::hash::{
        BuildHasher,
        Hasher,
        RandomState,
    };
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
