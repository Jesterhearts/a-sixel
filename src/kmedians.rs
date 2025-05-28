//! Uses k-medians to build a palette of colors. K-medians produces better
//! results than k-means, but is substantially slower.

use std::collections::HashSet;

use image::RgbImage;
use kiddo::{
    float::kdtree::KdTree,
    SquaredEuclidean,
};
use ordered_float::OrderedFloat;
use palette::{
    color_difference::EuclideanDistance,
    Lab,
};
use rayon::{
    iter::{
        IndexedParallelIterator,
        IntoParallelRefIterator,
        ParallelIterator,
    },
    slice::ParallelSliceMut,
};

use crate::{
    dither::Sierra,
    private,
    rgb_to_lab,
    PaletteBuilder,
    SixelEncoder,
};

pub type KMediansSixelEncoderMono<D = Sierra> = SixelEncoder<KMediansPaletteBuilder<2>, D>;
pub type KMediansSixelEncoder4<D = Sierra> = SixelEncoder<KMediansPaletteBuilder<4>, D>;
pub type KMediansSixelEncoder8<D = Sierra> = SixelEncoder<KMediansPaletteBuilder<8>, D>;
pub type KMediansSixelEncoder16<D = Sierra> = SixelEncoder<KMediansPaletteBuilder<16>, D>;
pub type KMediansSixelEncoder32<D = Sierra> = SixelEncoder<KMediansPaletteBuilder<32>, D>;
pub type KMediansSixelEncoder64<D = Sierra> = SixelEncoder<KMediansPaletteBuilder<64>, D>;
pub type KMediansSixelEncoder128<D = Sierra> = SixelEncoder<KMediansPaletteBuilder<128>, D>;
pub type KMediansSixelEncoder256<D = Sierra> = SixelEncoder<KMediansPaletteBuilder<256>, D>;

pub struct KMediansPaletteBuilder<const PALETTE_SIZE: usize>;

impl<const PALETTE_SIZE: usize> private::Sealed for KMediansPaletteBuilder<PALETTE_SIZE> {}

impl<const PALETTE_SIZE: usize> PaletteBuilder for KMediansPaletteBuilder<PALETTE_SIZE> {
    const PALETTE_SIZE: usize = PALETTE_SIZE;

    fn build_palette(image: &RgbImage) -> Vec<Lab> {
        let candidates = image
            .pixels()
            .copied()
            .map(rgb_to_lab)
            .map(|l| (l, 1.0))
            .collect::<Vec<_>>();

        parallel_kmedians::<PALETTE_SIZE>(&candidates)
    }
}

pub(crate) fn parallel_kmedians<const PALETTE_SIZE: usize>(candidates: &[(Lab, f32)]) -> Vec<Lab> {
    let mut centroids = KdTree::<_, _, 3, 257, u32>::with_capacity(PALETTE_SIZE);

    let center = candidates.par_iter().map(|(l, _)| *l).reduce(
        || <Lab>::new(0.0, 0.0, 0.0),
        |mut acc, color| {
            acc.l += color.l;
            acc.a += color.a;
            acc.b += color.b;
            acc
        },
    );
    let center = center / candidates.len() as f32;

    let (idx_furthest, _) = (0..candidates.len())
        .map(|idx| {
            let (color, _) = candidates[idx];
            let distance = color.distance_squared(center);
            (idx, distance)
        })
        .max_by_key(|(_, sum)| OrderedFloat(*sum))
        .unwrap();

    centroids.add(
        &[
            candidates[idx_furthest].0.l,
            candidates[idx_furthest].0.a,
            candidates[idx_furthest].0.b,
        ],
        0,
    );

    for cidx in 1..PALETTE_SIZE {
        let (furthest, _) = candidates
            .par_iter()
            .copied()
            .max_by_key(|(c, _)| {
                let nearest = centroids.nearest_one::<SquaredEuclidean>(&[c.l, c.a, c.b]);
                OrderedFloat(nearest.distance)
            })
            .unwrap();

        centroids.add(&[furthest.l, furthest.a, furthest.b], cidx as u32);
    }

    let mut cluster_assignments = vec![[false; PALETTE_SIZE]; candidates.len()];
    candidates
        .par_iter()
        .copied()
        .zip(&mut cluster_assignments)
        .for_each(|((color, _), slot)| {
            let nearest = centroids.nearest_one::<SquaredEuclidean>(&[color.l, color.a, color.b]);
            slot[nearest.item as usize] = true;
        });

    for _ in 0..100 {
        centroids = KdTree::<_, _, 3, 257, u32>::with_capacity(PALETTE_SIZE);

        for idx in 0..PALETTE_SIZE {
            let mut ls = candidates
                .par_iter()
                .zip(&cluster_assignments)
                .filter_map(
                    |(cw, assignments)| {
                        if assignments[idx] {
                            Some(cw)
                        } else {
                            None
                        }
                    },
                )
                .collect::<Vec<_>>();

            ls.par_sort_unstable_by_key(|(color, w)| (OrderedFloat(color.l), OrderedFloat(*w)));
            let w_sum = ls.par_iter().map(|(_, w)| *w).sum::<f32>();
            let median_l = ls
                .iter()
                .scan(0.0, |sum, (color, w)| {
                    if *sum >= w_sum / 2.0 {
                        None
                    } else {
                        *sum += w;
                        Some(color)
                    }
                })
                .last()
                .unwrap()
                .l;

            ls.par_sort_unstable_by_key(|(color, w)| (OrderedFloat(color.a), OrderedFloat(*w)));
            let median_a = ls
                .iter()
                .scan(0.0, |sum, (color, w)| {
                    if *sum >= w_sum / 2.0 {
                        None
                    } else {
                        *sum += w;
                        Some(color)
                    }
                })
                .last()
                .unwrap()
                .a;

            ls.par_sort_unstable_by_key(|(color, w)| (OrderedFloat(color.b), OrderedFloat(*w)));
            let median_b = ls
                .iter()
                .scan(0.0, |sum, (color, w)| {
                    if *sum >= w_sum / 2.0 {
                        None
                    } else {
                        *sum += w;
                        Some(color)
                    }
                })
                .last()
                .unwrap()
                .b;

            centroids.add(&[median_l, median_a, median_b], idx as u32);
        }

        let shifts = candidates
            .par_iter()
            .copied()
            .zip(&mut cluster_assignments)
            .map(|((color, _), slot)| {
                let nearest =
                    centroids.nearest_one::<SquaredEuclidean>(&[color.l, color.a, color.b]);
                if slot[nearest.item as usize] {
                    return false;
                }

                slot.fill(false);
                slot[nearest.item as usize] = true;

                true
            })
            .filter(|shift| *shift)
            .count();

        if shifts == 0 {
            break;
        }
    }

    centroids
        .iter()
        .map(|(_, centroid)| {
            [
                OrderedFloat(centroid[0]),
                OrderedFloat(centroid[1]),
                OrderedFloat(centroid[2]),
            ]
        })
        .collect::<HashSet<_>>()
        .into_iter()
        .map(|[l, a, b]| Lab::new(*l, *a, *b))
        .collect()
}
