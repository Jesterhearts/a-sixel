//! Uses k-medians to build a palette of colors. K-medians produces better
//! results than k-means, but is substantially slower.

use std::{
    array,
    collections::HashSet,
    sync::atomic::{
        AtomicBool,
        Ordering,
    },
};

use dashmap::DashSet;
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
    const NAME: &'static str = "K-Medians";
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

    let mut medians = [[0.0; 3]; PALETTE_SIZE];
    let (idx_furthest, _) = (0..candidates.len())
        .map(|idx| {
            let (color, _) = candidates[idx];
            let distance = color.distance_squared(center);
            (idx, distance)
        })
        .max_by_key(|(_, sum)| OrderedFloat(*sum))
        .unwrap();

    medians[0] = [
        candidates[idx_furthest].0.l,
        candidates[idx_furthest].0.a,
        candidates[idx_furthest].0.b,
    ];
    centroids.add(
        &[
            candidates[idx_furthest].0.l,
            candidates[idx_furthest].0.a,
            candidates[idx_furthest].0.b,
        ],
        0,
    );

    for (cidx, medians) in medians.iter_mut().enumerate().skip(1) {
        let (furthest, _) = candidates
            .par_iter()
            .copied()
            .max_by_key(|(ref c, _)| {
                let nearest = centroids.nearest_one::<SquaredEuclidean>(&[c.l, c.a, c.b]);
                OrderedFloat(nearest.distance)
            })
            .unwrap();

        *medians = [furthest.l, furthest.a, furthest.b];
        centroids.add(&[furthest.l, furthest.a, furthest.b], cidx as u32);
    }

    let mut cluster_assignments = array::from_fn::<_, PALETTE_SIZE, _>(|_| DashSet::<KLabW>::new());
    candidates.par_iter().copied().for_each(|(color, w)| {
        let nearest = centroids.nearest_one::<SquaredEuclidean>(&[color.l, color.a, color.b]);
        cluster_assignments[nearest.item as usize].insert((color, w).into());
    });

    for _ in 0..100 {
        centroids = KdTree::<_, _, 3, 257, u32>::with_capacity(PALETTE_SIZE);

        cluster_assignments
            .par_iter()
            .zip(&mut medians)
            .for_each(|(set, medians)| {
                let mut ls = set
                    .iter()
                    .map(|klw| (*klw).into())
                    .collect::<Vec<(Lab, f32)>>();

                if ls.is_empty() {
                    return;
                }

                let w_sum = ls.iter().map(|(_, w)| *w).sum::<f32>();
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

                *medians = [median_l, median_a, median_b];
            });

        for (idx, medians) in medians.into_iter().enumerate() {
            centroids.add(&medians, idx as u32);
        }

        let old_assignments = cluster_assignments;
        cluster_assignments = array::from_fn(|_| DashSet::<KLabW>::new());
        let had_swap = AtomicBool::new(false);
        candidates.par_iter().copied().for_each(|(color, w)| {
            let nearest = centroids.nearest_one::<SquaredEuclidean>(&[color.l, color.a, color.b]);
            let kw = KLabW::from((color, w));

            cluster_assignments[nearest.item as usize].insert(kw);

            let swapped = !old_assignments[nearest.item as usize].contains(&kw);
            had_swap.fetch_or(swapped, Ordering::Relaxed);
        });

        if !had_swap.load(Ordering::Relaxed) {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
struct KLabW {
    l: OrderedFloat<f32>,
    a: OrderedFloat<f32>,
    b: OrderedFloat<f32>,
    w: OrderedFloat<f32>,
}

impl From<(Lab, f32)> for KLabW {
    fn from((lab, w): (Lab, f32)) -> Self {
        Self {
            l: OrderedFloat(lab.l),
            a: OrderedFloat(lab.a),
            b: OrderedFloat(lab.b),
            w: OrderedFloat(w),
        }
    }
}

impl From<KLabW> for (Lab, f32) {
    fn from(klw: KLabW) -> Self {
        (Lab::new(klw.l.0, klw.a.0, klw.b.0), klw.w.0)
    }
}
