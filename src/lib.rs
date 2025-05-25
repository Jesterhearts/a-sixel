use core::f32;
use std::{
    collections::HashSet,
    f32::consts::{
        E,
        PI,
    },
    fmt::Write,
};

use image::{
    Rgb,
    RgbImage,
};
use kiddo::{
    SquaredEuclidean,
    float::kdtree::KdTree,
    traits::DistanceMetric,
};
use libblur::{
    AnisotropicRadius,
    BlurImageMut,
    FastBlurChannels,
    ThreadingPolicy,
    stack_blur_f32,
};
use ordered_float::NotNan;
use palette::{
    Hsl,
    IntoColor,
    Lab,
    color_difference::EuclideanDistance,
    encoding::Srgb,
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
    FftPlanner,
    num_complex::Complex,
    num_traits::Zero,
};
use sobol_burley::sample_4d;

struct SixelRow<'c> {
    committed: &'c mut String,
    pending: char,
    count: usize,
}

impl<'c> SixelRow<'c> {
    fn new(builder: &'c mut String, color: usize) -> Self {
        builder
            .write_fmt(format_args!("#{color}"))
            .expect("Failed to write color selector");

        Self {
            committed: builder,
            pending: num2six(0),
            count: 0,
        }
    }

    fn push(&mut self, ch: char) {
        if ch == self.pending {
            self.count += 1;
        } else {
            self.commit();
            self.pending = ch;
            self.count = 1;
        }
    }

    fn commit(&mut self) {
        if self.count > 3 {
            self.committed
                .write_fmt(format_args!("!{}{}", self.count, self.pending))
                .expect("Failed to write to string");
        } else {
            for _ in 0..self.count {
                self.committed.push(self.pending);
            }
        }
    }

    fn finalize(mut self) {
        self.commit();
        self.committed.push('$');
    }
}

mod private {
    pub trait Sealed {}
}

/// https://tannerhelland.com/2012/12/28/dithering-eleven-algorithms-source-code.html
pub trait Dither: private::Sealed {
    const KERNEL: &[(isize, isize, f32)];
    const DIV: f32;

    fn dither_and_palettize(image: &RgbImage, in_palette: &[Lab]) -> Vec<usize> {
        let pixels = image.pixels().copied().map(rgb_to_lab).collect::<Vec<_>>();

        let mut palette = KdTree::<_, _, 3, 32, u32>::with_capacity(in_palette.len());

        for (idx, color) in in_palette.iter().enumerate() {
            palette.add(color.as_ref(), idx);
        }

        let mut result = vec![0; image.width() as usize * image.height() as usize];

        let mut spills = vec![[0.0; 3]; image.width() as usize * image.height() as usize];

        for (idx, p) in pixels.iter().copied().enumerate() {
            let pixel = <Lab>::new(
                p.l + spills[idx][0],
                p.a + spills[idx][1],
                p.b + spills[idx][2],
            );

            let palette_idx =
                palette.nearest_one::<kiddo::SquaredEuclidean>(&[pixel.l, pixel.a, pixel.b]);

            let error0 = pixel.l - in_palette[palette_idx.item].l;
            let error1 = pixel.a - in_palette[palette_idx.item].a;
            let error2 = pixel.b - in_palette[palette_idx.item].b;
            let spill = [error0, error1, error2];

            result[idx] = palette_idx.item;

            for (dx, dy, m) in Self::KERNEL {
                let x = idx as isize % image.width() as isize + dx;
                let y = idx as isize / image.width() as isize + dy;
                if x < 0 || y < 0 || x >= image.width() as isize || y >= image.height() as isize {
                    continue;
                }

                let jdx = y * image.width() as isize + x;
                let target = &mut spills[jdx as usize];
                *target = [
                    target[0] + (spill[0] * m) / Self::DIV / 2.0,
                    target[1] + (spill[1] * m) / Self::DIV / 2.0,
                    target[2] + (spill[2] * m) / Self::DIV / 2.0,
                ];
            }
        }

        result
    }
}

pub struct NoDither;
impl private::Sealed for NoDither {}
impl Dither for NoDither {
    const DIV: f32 = 1.0;
    const KERNEL: &[(isize, isize, f32)] = &[];
}

pub struct Sierra;
impl private::Sealed for Sierra {}
impl Dither for Sierra {
    const DIV: f32 = 32.0;
    // _ _ X 5 3
    // 2 4 5 4 2
    // _ 2 3 2 _
    const KERNEL: &[(isize, isize, f32)] = &[
        (1, 0, 5.0),
        (2, 0, 3.0),
        //
        (-2, 1, 2.0),
        (-1, 1, 4.0),
        (0, 1, 5.0),
        (1, 1, 4.0),
        (2, 1, 2.0),
        //
        (-1, 2, 2.0),
        (0, 2, 3.0),
        (1, 2, 2.0),
    ];
}

pub struct Sierra2;
impl private::Sealed for Sierra2 {}
impl Dither for Sierra2 {
    const DIV: f32 = 16.0;
    // _ _ X 4 3
    // 1 2 3 2 1
    // _ 1 2 1 _
    const KERNEL: &[(isize, isize, f32)] = &[
        //x y  m
        (1, 0, 4.0),
        (2, 0, 3.0),
        //
        (-2, 1, 1.0),
        (-1, 1, 2.0),
        (0, 1, 3.0),
        (1, 1, 2.0),
        (2, 1, 1.0),
    ];
}

pub struct SierraLite;
impl private::Sealed for SierraLite {}
impl Dither for SierraLite {
    const DIV: f32 = 4.0;
    //  _ X 2
    //  1 1 _
    const KERNEL: &[(isize, isize, f32)] = &[
        //x y  m
        (1, 0, 2.0),
        //
        (-1, 1, 1.0),
        (0, 1, 1.0),
    ];
}

pub struct FloydSteinberg;
impl private::Sealed for FloydSteinberg {}
impl Dither for FloydSteinberg {
    const DIV: f32 = 16.0;
    // _ X 7
    // 3 5 3
    const KERNEL: &[(isize, isize, f32)] = &[
        //x y  m
        (1, 0, 7.0),
        //
        (-1, 1, 3.0),
        (0, 1, 5.0),
        (1, 1, 1.0),
    ];
}

pub struct JJN;
impl private::Sealed for JJN {}
impl Dither for JJN {
    const DIV: f32 = 48.0;
    // _ _ X 7 5
    // 3 5 7 5 3
    // 1 2 5 3 1
    const KERNEL: &[(isize, isize, f32)] = &[
        //x y  m
        (1, 0, 7.0),
        (2, 0, 5.0),
        //
        (-2, 1, 3.0),
        (-1, 1, 5.0),
        (0, 1, 7.0),
        (1, 1, 5.0),
        (2, 1, 3.0),
        //
        (-2, 2, 1.0),
        (-1, 2, 2.0),
        (0, 2, 5.0),
        (1, 2, 3.0),
        (2, 2, 1.0),
    ];
}

pub struct Stucki;
impl private::Sealed for Stucki {}
impl Dither for Stucki {
    const DIV: f32 = 42.0;
    // _ _ X 8 4
    // 2 4 8 4 2
    // 1 2 4 2 1
    const KERNEL: &[(isize, isize, f32)] = &[
        //x y  m
        (1, 0, 8.0),
        (2, 0, 4.0),
        //
        (-2, 1, 2.0),
        (-1, 1, 4.0),
        (0, 1, 8.0),
        (1, 1, 4.0),
        (2, 1, 2.0),
        //
        (-2, 2, 1.0),
        (-1, 2, 2.0),
        (0, 2, 4.0),
        (1, 2, 2.0),
        (2, 2, 1.0),
    ];
}

pub struct Atkinson;
impl private::Sealed for Atkinson {}
impl Dither for Atkinson {
    const DIV: f32 = 8.0;
    // _ X 1 1
    // 1 1 1 _
    // _ 1 _ _
    const KERNEL: &[(isize, isize, f32)] = &[
        //x y  m
        (1, 0, 1.0),
        (2, 0, 1.0),
        //
        (-1, 1, 1.0),
        (0, 1, 1.0),
        (1, 1, 1.0),
        //
        (0, 2, 1.0),
    ];
}

pub struct Burkes;
impl private::Sealed for Burkes {}
impl Dither for Burkes {
    const DIV: f32 = 32.0;
    // _ _ X 8 4
    // 2 4 8 4 2
    const KERNEL: &[(isize, isize, f32)] = &[
        //x y  m
        (1, 0, 8.0),
        (2, 0, 4.0),
        //
        (-2, 1, 2.0),
        (-1, 1, 4.0),
        (0, 1, 8.0),
        (1, 1, 4.0),
        (2, 1, 2.0),
    ];
}

pub trait PaletteBuilder: private::Sealed {
    const PALETTE_SIZE: usize;

    fn build_palette(image: &RgbImage) -> Vec<Lab>;
}

pub struct MedianCutPaletteBuilder<const PALETTE_SIZE: usize = 256>;
impl<const PALETTE_SIZE: usize> private::Sealed for MedianCutPaletteBuilder<PALETTE_SIZE> {}
impl<const PALETTE_SIZE: usize> PaletteBuilder for MedianCutPaletteBuilder<PALETTE_SIZE> {
    const PALETTE_SIZE: usize = PALETTE_SIZE;

    fn build_palette(image: &RgbImage) -> Vec<Lab> {
        let pixels = image.pixels().copied().map(rgb_to_lab).collect::<Vec<_>>();

        let mut buckets = Vec::with_capacity(PALETTE_SIZE);
        buckets.push(pixels);
        let mut bucket_stats = vec![None; PALETTE_SIZE + 1];

        for _ in 0..PALETTE_SIZE - 1 {
            let mut best_bucket = 0;
            let mut max_idx = 0;
            let mut max_range = 0.0;
            for (idx, (candidates, stats)) in
                buckets.iter().zip(bucket_stats.iter_mut()).enumerate()
            {
                let (min, max) = if let Some((min, max)) = *stats {
                    (min, max)
                } else {
                    let (min, max) = candidates.iter().copied().fold(
                        (
                            <Lab>::new(f32::MAX, f32::MAX, f32::MAX),
                            <Lab>::new(f32::MIN, f32::MIN, f32::MIN),
                        ),
                        |(min, max), color| {
                            (
                                Lab::new(
                                    min.l.min(color.l),
                                    min.a.min(color.a),
                                    min.b.min(color.b),
                                ),
                                Lab::new(
                                    max.l.max(color.l),
                                    max.a.max(color.a),
                                    max.b.max(color.b),
                                ),
                            )
                        },
                    );
                    *stats = Some((min, max));
                    (min, max)
                };

                let range = [
                    (max.l - min.l) / (<Lab>::max_l() - <Lab>::min_l()),
                    (max.a - min.a) / (<Lab>::max_a() - <Lab>::min_a()),
                    (max.b - min.b) / (<Lab>::max_b() - <Lab>::min_b()),
                ];
                let max_range_idx = range
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, diff)| NotNan::new(**diff).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();
                if range[max_range_idx] > max_range {
                    best_bucket = idx;
                    max_idx = max_range_idx;
                    max_range = range[max_range_idx];
                }
            }

            let candidates = &mut buckets[best_bucket];
            candidates.sort_by(|a, b| match max_idx {
                0 => a.l.total_cmp(&b.l),
                1 => a.a.total_cmp(&b.a),
                2 => a.b.total_cmp(&b.b),
                _ => unreachable!(),
            });

            let median_idx = candidates.len() / 2;
            bucket_stats[best_bucket] = None;
            let new_candidates = candidates.split_off(median_idx);
            bucket_stats[buckets.len()] = None;
            buckets.push(new_candidates);
        }

        buckets
            .into_iter()
            .map(|b| {
                let b_len = b.len();
                b.into_iter()
                    .fold(<Lab>::new(0.0, 0.0, 0.0), |mut acc, color| {
                        acc.l += color.l;
                        acc.a += color.a;
                        acc.b += color.b;
                        acc
                    })
                    / b_len as f32
            })
            .collect()
    }
}

pub struct ADUPaletteBuilder<
    const PALETTE_SIZE: usize = 256,
    const THETA: usize = 16,
    const STEPS: usize = 4096,
    const GAMMA_DIV: usize = 8,
>;

impl<const PALETTE_SIZE: usize, const THETA: usize, const STEPS: usize, const GAMMA_DIV: usize>
    private::Sealed for ADUPaletteBuilder<PALETTE_SIZE, THETA, STEPS, GAMMA_DIV>
{
}
impl<const PALETTE_SIZE: usize, const THETA: usize, const STEPS: usize, const GAMMA_DIV: usize>
    PaletteBuilder for ADUPaletteBuilder<PALETTE_SIZE, THETA, STEPS, GAMMA_DIV>
{
    const PALETTE_SIZE: usize = PALETTE_SIZE;

    fn build_palette(image: &RgbImage) -> Vec<Lab> {
        let gamma: f32 = 1.0 / (GAMMA_DIV as f32);

        let candidates = image.pixels().copied().map(rgb_to_lab).collect::<Vec<_>>();

        let centroid = candidates.par_iter().copied().reduce(
            || <Lab>::new(0.0, 0.0, 0.0),
            |mut acc, color| {
                acc.l += color.l;
                acc.a += color.a;
                acc.b += color.b;
                acc
            },
        ) / candidates.len() as f32;

        let mut palette = [centroid; PALETTE_SIZE];

        let mut tree = KdTree::<_, _, 3, PALETTE_SIZE, u32>::with_capacity(PALETTE_SIZE);
        tree.add(&[palette[0].l, palette[0].a, palette[0].b], 0);

        let mut next_idx = 1;

        let mut wc = [0; PALETTE_SIZE];

        let candidates = (0..STEPS as u32 / 4)
            .flat_map(|idx| {
                let [x, y, z, w] = sample_4d(idx % (1 << 16), 0, idx / (1 << 16));
                [
                    candidates[(x * candidates.len() as f32) as usize],
                    candidates[(y * candidates.len() as f32) as usize],
                    candidates[(z * candidates.len() as f32) as usize],
                    candidates[(w * candidates.len() as f32) as usize],
                ]
            })
            .collect::<Vec<_>>();

        for candidate in candidates {
            let winner =
                tree.nearest_one::<SquaredEuclidean>(&[candidate.l, candidate.a, candidate.b]);

            tree.remove(
                &[
                    palette[winner.item].l,
                    palette[winner.item].a,
                    palette[winner.item].b,
                ],
                winner.item,
            );

            palette[winner.item].l += (candidate.l - palette[winner.item].l) * gamma;
            palette[winner.item].a += (candidate.a - palette[winner.item].a) * gamma;
            palette[winner.item].b += (candidate.b - palette[winner.item].b) * gamma;

            tree.add(
                &[
                    palette[winner.item].l,
                    palette[winner.item].a,
                    palette[winner.item].b,
                ],
                winner.item,
            );

            wc[winner.item] += 1;

            if wc[winner.item] >= THETA && next_idx < PALETTE_SIZE {
                tree.add(&[candidate.l, candidate.a, candidate.b], next_idx);

                wc[winner.item] = 0;
                wc[next_idx] = 0;
                next_idx += 1;
            }
        }

        palette
            .into_iter()
            .map(|lab| {
                [
                    NotNan::new(lab.l).unwrap(),
                    NotNan::new(lab.a).unwrap(),
                    NotNan::new(lab.b).unwrap(),
                ]
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .map(|[l, a, b]| Lab::new(*l, *a, *b))
            .collect::<Vec<_>>()
    }
}

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

        let centroid = pixels.par_iter().copied().reduce(
            || <Lab>::new(0.0, 0.0, 0.0),
            |mut acc, color| {
                acc.l += color.l;
                acc.a += color.a;
                acc.b += color.b;
                acc
            },
        ) / pixels.len() as f32;

        let mut dists = vec![0.0f32; pixels.len()];
        pixels
            .par_iter()
            .copied()
            .zip(&mut dists)
            .for_each(|(lab, dest)| {
                *dest = lab.distance(centroid);
            });

        let max_dist = dists.par_iter().copied().reduce(|| f32::MIN, f32::max);
        let min_dist = dists.par_iter().copied().reduce(|| f32::MAX, f32::min);
        dists.par_iter_mut().for_each(|d| {
            *d = (*d - min_dist) / (max_dist - min_dist).max(f32::EPSILON);
        });

        #[cfg(feature = "dump_dist_saliency")]
        {
            let mut quant_dists = vec![0; pixels.len()];
            dists
                .par_iter()
                .copied()
                .zip(&mut quant_dists)
                .for_each(|(d, dest)| {
                    *dest = (d * u8::MAX as f32).round() as u8;
                });

            dump_intermediate("dists", &quant_dists, image_width, image_height);
        }

        let avg_dist = dists.par_iter().copied().sum::<f32>() / dists.len() as f32;

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

                *dest = sum / count as f32
            });

        let min_local_dist = local_dists
            .par_iter()
            .copied()
            .reduce(|| f32::MAX, f32::min);
        let max_local_dist = local_dists
            .par_iter()
            .copied()
            .reduce(|| f32::MIN, f32::max);

        local_dists.par_iter_mut().for_each(|d| {
            *d = (*d - min_local_dist) / (max_local_dist - min_local_dist).max(f32::EPSILON);
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

        let avg_local_dist =
            local_dists.par_iter().copied().sum::<f32>() / local_dists.len() as f32;

        let mut candidates = vec![(<Lab>::new(0.0, 0.0, 0.0), 0.0); pixels.len()];
        pixels
            .into_par_iter()
            .zip(l_saliency.spectral_residual)
            .zip(l_saliency.phase_spectrum)
            .zip(l_saliency.amplitude_spectrum)
            .zip(a_saliency.spectral_residual)
            .zip(a_saliency.phase_spectrum)
            .zip(a_saliency.amplitude_spectrum)
            .zip(b_saliency.spectral_residual)
            .zip(b_saliency.phase_spectrum)
            .zip(b_saliency.amplitude_spectrum)
            .zip(local_dists)
            .zip(&mut candidates)
            .for_each(
                |(
                    ((((((((((lab, l_sr), l_p), l_a), a_sr), a_p), a_a), b_sr), b_p), b_a), local),
                    dest,
                )| {
                    let outliers = l_sr.max(a_sr).max(b_sr);
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

        candidates.par_sort_by_key(|(_, w)| NotNan::new(*w).unwrap());

        let mut palette = candidates[candidates.len() - 1..].to_vec();

        let mut tree = KdTree::<_, _, 5, 257, u32>::with_capacity(PALETTE_SIZE);

        let weight_term = (-avg_dist).exp();
        for (idx, (lab, w)) in palette.iter().copied().enumerate() {
            tree.add(&[lab.l, lab.a, lab.b, w, weight_term], idx);
        }

        let strength = 1.0 - 0.3 * avg_local_dist - 0.7 * avg_dist * avg_local_dist;
        let skew = |x: f32| x.powf(1.0 / E.powf(strength));

        let wc_thresh = 2;
        let alpha = (1.0 + avg_local_dist - avg_dist)
            .abs()
            .sqrt()
            .clamp(0.125, 0.5);
        let mut wc = [0; PALETTE_SIZE];

        const MIN_STEPS: u32 = 1 << 17;
        let mut idx = 0;
        while idx < MIN_STEPS / 4 || palette.len() < PALETTE_SIZE {
            let samples = sample_4d(idx % (1 << 16), 0, idx / (1 << 16));
            idx += 1;
            for candidate in samples {
                let candidate = skew(candidate);
                let (candidate, w) = candidates[(candidate * candidates.len() as f32) as usize];

                let winner = tree.nearest_one::<LabWDist>(&[
                    candidate.l,
                    candidate.a,
                    candidate.b,
                    w,
                    weight_term,
                ]);

                tree.remove(
                    &[
                        palette[winner.item].0.l,
                        palette[winner.item].0.a,
                        palette[winner.item].0.b,
                        palette[winner.item].1,
                        weight_term,
                    ],
                    winner.item,
                );

                palette[winner.item].0.l += (candidate.l - palette[winner.item].0.l) * alpha;
                palette[winner.item].0.a += (candidate.a - palette[winner.item].0.a) * alpha;
                palette[winner.item].0.b += (candidate.b - palette[winner.item].0.b) * alpha;
                palette[winner.item].1 += (w - palette[winner.item].1) * alpha;

                tree.add(
                    &[
                        palette[winner.item].0.l,
                        palette[winner.item].0.a,
                        palette[winner.item].0.b,
                        palette[winner.item].1,
                        weight_term,
                    ],
                    winner.item,
                );

                wc[winner.item] += 1;

                if wc[winner.item] >= wc_thresh && palette.len() < PALETTE_SIZE {
                    tree.add(
                        &[candidate.l, candidate.a, candidate.b, w, weight_term],
                        palette.len(),
                    );

                    wc[winner.item] = 0;
                    wc[palette.len()] = 0;
                    palette.push((candidate, w));
                }
            }
        }

        palette
            .into_iter()
            .map(|(lab, _)| {
                [
                    NotNan::new(lab.l).unwrap(),
                    NotNan::new(lab.a).unwrap(),
                    NotNan::new(lab.b).unwrap(),
                ]
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .map(|[l, a, b]| Lab::new(*l, *a, *b))
            .collect::<Vec<_>>()
    }
}

struct Saliency {
    spectral_residual: Vec<f32>,
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

    let asr = {
        let mut filtered = vec![0.0f32; phase.len()];
        let max_amplitude = amplitude.par_iter().copied().reduce(|| f32::MIN, f32::max);
        let min_amplitude = amplitude.par_iter().copied().reduce(|| f32::MAX, f32::min);
        amplitude
            .par_iter()
            .copied()
            .zip(&mut filtered)
            .for_each(|(a, dest)| {
                *dest = (a - min_amplitude) / (max_amplitude - min_amplitude).max(f32::EPSILON);
            });

        filtered.par_iter_mut().for_each(|a: &mut f32| {
            *a = ((*a * PI / 2.0).cos().powi(5) + (*a * PI / 2.0).sin().powi(15))
                * (max_amplitude - min_amplitude)
                + min_amplitude;
        });

        let mut ifft_buffer = vec![Complex::zero(); phase.len()];
        filtered
            .into_par_iter()
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
                .zip(phase)
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

    Saliency {
        spectral_residual: sr,
        phase_spectrum: pft,
        amplitude_spectrum: asr,
    }
}

struct LabWDist;

impl DistanceMetric<f32, 5> for LabWDist {
    fn dist(a: &[f32; 5], b: &[f32; 5]) -> f32 {
        let la = <Lab>::new(a[0], a[1], a[2]);
        let lb = <Lab>::new(b[0], b[1], b[2]);
        la.distance_squared(lb) + (a[3] - b[3]).abs() * a[4]
    }

    fn dist1(a: f32, b: f32) -> f32 {
        (a - b).abs()
    }
}

const fn num2six(num: u8) -> char {
    (0x3f + num) as char
}

pub struct SixelEncoder<P: PaletteBuilder = FocalPaletteBuilder, D: Dither = Sierra> {
    _p: std::marker::PhantomData<P>,
    _d: std::marker::PhantomData<D>,
}

pub type ADUSixelEncoder8<D = Sierra> = SixelEncoder<ADUPaletteBuilder<8, 1, { 1 << 17 }>, D>;
pub type ADUSixelEncoder16<D = Sierra> = SixelEncoder<ADUPaletteBuilder<16, 1, { 1 << 17 }>, D>;
pub type ADUSixelEncoder32<D = Sierra> = SixelEncoder<ADUPaletteBuilder<32, 2, { 1 << 17 }>, D>;
pub type ADUSixelEncoder64<D = Sierra> = SixelEncoder<ADUPaletteBuilder<64, 4, { 1 << 17 }>, D>;
pub type ADUSixelEncoder128<D = Sierra> = SixelEncoder<ADUPaletteBuilder<128, 8, { 1 << 17 }>, D>;
pub type ADUSixelEncoder256<D = Sierra> = SixelEncoder<ADUPaletteBuilder<256, 16, { 1 << 17 }>, D>;
pub type ADUSixelEncoder<D = Sierra> = ADUSixelEncoder256<D>;

pub type FocalSixelEncoderMono<D = Sierra> = SixelEncoder<FocalPaletteBuilder<2>, D>;
pub type FocalSixelEncoder4<D = Sierra> = SixelEncoder<FocalPaletteBuilder<4>, D>;
pub type FocalSixelEncoder8<D = Sierra> = SixelEncoder<FocalPaletteBuilder<8>, D>;
pub type FocalSixelEncoder16<D = Sierra> = SixelEncoder<FocalPaletteBuilder<16>, D>;
pub type FocalSixelEncoder32<D = Sierra> = SixelEncoder<FocalPaletteBuilder<32>, D>;
pub type FocalSixelEncoder64<D = Sierra> = SixelEncoder<FocalPaletteBuilder<64>, D>;
pub type FocalSixelEncoder128<D = Sierra> = SixelEncoder<FocalPaletteBuilder<128>, D>;
pub type FocalSixelEncoder256<D = Sierra> = SixelEncoder<FocalPaletteBuilder<256>, D>;
pub type FocalSixelEncoder<D = Sierra> = FocalSixelEncoder256<D>;

pub type MedianCutSixelEncoderMono<D = Sierra> = SixelEncoder<MedianCutPaletteBuilder<2>, D>;
pub type MedianCutSixelEncoder4<D = Sierra> = SixelEncoder<MedianCutPaletteBuilder<4>, D>;
pub type MedianCutSixelEncoder8<D = Sierra> = SixelEncoder<MedianCutPaletteBuilder<8>, D>;
pub type MedianCutSixelEncoder16<D = Sierra> = SixelEncoder<MedianCutPaletteBuilder<16>, D>;
pub type MedianCutSixelEncoder32<D = Sierra> = SixelEncoder<MedianCutPaletteBuilder<32>, D>;
pub type MedianCutSixelEncoder64<D = Sierra> = SixelEncoder<MedianCutPaletteBuilder<64>, D>;
pub type MedianCutSixelEncoder128<D = Sierra> = SixelEncoder<MedianCutPaletteBuilder<128>, D>;
pub type MedianCutSixelEncoder256<D = Sierra> = SixelEncoder<MedianCutPaletteBuilder<256>, D>;
pub type MedianCutSixelEncoder<D = Sierra> = MedianCutSixelEncoder256<D>;

impl<P: PaletteBuilder, D: Dither> SixelEncoder<P, D> {
    pub fn encode(image: RgbImage) -> String {
        let palette = P::build_palette(&image);
        let mut sixel_string = r#"Pq"1;1;"#.to_string();
        sixel_string
            .write_fmt(format_args!("{};{}", image.height(), image.width()))
            .expect("Failed to write sixel bounds");

        for (i, lab) in palette.iter().copied().enumerate() {
            let hsl: Hsl = lab.into_color();
            // This may be a windows specific bug, but hue is offset by 120 degrees.
            let deg = (hsl.hue.into_positive_degrees().round() as u16 + 120) % 360;

            sixel_string
                .write_fmt(format_args!(
                    "#{i};1;{deg};{};{}",
                    (hsl.lightness * 100.0).round() as u8,
                    (hsl.saturation * 100.0).round() as u8,
                ))
                .expect("Failed to palette entry");
        }

        let paletted_pixels = D::dither_and_palettize(&image, &palette);

        let rows: Vec<&[usize]> = paletted_pixels
            .chunks(image.width() as usize)
            .collect::<Vec<_>>();

        let mut row_palette = vec![false; P::PALETTE_SIZE];
        for stack in rows.chunks(6) {
            row_palette.fill(false);
            for idx in stack_iter(stack).flat_map(|(((((zero, one), two), three), four), five)| {
                std::iter::once(zero)
                    .chain(one)
                    .chain(two)
                    .chain(three)
                    .chain(four)
                    .chain(five)
            }) {
                row_palette[idx] = true;
            }

            for (color, _) in row_palette.iter().copied().enumerate().filter(|(_, v)| *v) {
                let mut stack_string = SixelRow::new(&mut sixel_string, color);
                for (((((zero, one), two), three), four), five) in stack_iter(stack) {
                    let bits = (zero == color) as u8
                        | ((one == Some(color)) as u8) << 1
                        | ((two == Some(color)) as u8) << 2
                        | ((three == Some(color)) as u8) << 3
                        | ((four == Some(color)) as u8) << 4
                        | ((five == Some(color)) as u8) << 5;
                    let char = num2six(bits);
                    stack_string.push(char);
                }
                stack_string.finalize();
            }
            sixel_string.push('-');
        }

        sixel_string.push_str(r#"\"#);
        sixel_string
    }
}

type StackTuple = (
    (
        (((usize, Option<usize>), Option<usize>), Option<usize>),
        Option<usize>,
    ),
    Option<usize>,
);

fn stack_iter(stack: &[&[usize]]) -> impl Iterator<Item = StackTuple> {
    stack
        .first()
        .into_iter()
        .cloned()
        .flatten()
        .copied()
        .zip(
            stack
                .get(1)
                .into_iter()
                .cloned()
                .flatten()
                .copied()
                .map(Some)
                .chain(std::iter::repeat(None)),
        )
        .zip(
            stack
                .get(2)
                .into_iter()
                .cloned()
                .flatten()
                .copied()
                .map(Some)
                .chain(std::iter::repeat(None)),
        )
        .zip(
            stack
                .get(3)
                .into_iter()
                .cloned()
                .flatten()
                .copied()
                .map(Some)
                .chain(std::iter::repeat(None)),
        )
        .zip(
            stack
                .get(4)
                .into_iter()
                .cloned()
                .flatten()
                .copied()
                .map(Some)
                .chain(std::iter::repeat(None)),
        )
        .zip(
            stack
                .get(5)
                .into_iter()
                .cloned()
                .flatten()
                .copied()
                .map(Some)
                .chain(std::iter::repeat(None)),
        )
}

fn rgb_to_lab(Rgb([r, g, b]): Rgb<u8>) -> Lab {
    palette::rgb::Rgb::<Srgb, _>::new(r, g, b)
        .into_format::<f32>()
        .into_color()
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
