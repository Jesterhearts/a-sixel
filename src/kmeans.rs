use std::{
    array,
    collections::HashSet,
    sync::atomic::Ordering,
};

use atomic_float::AtomicF32;
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
use rayon::iter::{
    IndexedParallelIterator,
    IntoParallelRefIterator,
    ParallelIterator,
};

use crate::{
    private,
    rgb_to_lab,
    PaletteBuilder,
};

/// Performs K-Means clustering on the image's pixels to build a palette.
pub struct KMeansPaletteBuilder<const PALETTE_SIZE: usize>;

impl<const PALETTE_SIZE: usize> private::Sealed for KMeansPaletteBuilder<PALETTE_SIZE> {}

impl<const PALETTE_SIZE: usize> PaletteBuilder for KMeansPaletteBuilder<PALETTE_SIZE> {
    const PALETTE_SIZE: usize = PALETTE_SIZE;

    fn build_palette(image: &RgbImage) -> Vec<Lab> {
        let candidates = image
            .pixels()
            .copied()
            .map(rgb_to_lab)
            .map(|c| (c, 1.0))
            .collect::<Vec<_>>();

        parallel_kmeans::<PALETTE_SIZE>(&candidates)
    }
}

pub(crate) fn parallel_kmeans<const PALETTE_SIZE: usize>(candidates: &[(Lab, f32)]) -> Vec<Lab> {
    let mut centroids = KdTree::<_, _, 3, 257, u32>::with_capacity(PALETTE_SIZE);

    let (center, weight) = candidates.par_iter().copied().reduce(
        || (<Lab>::new(0.0, 0.0, 0.0), 0.0),
        |mut acc, color| {
            acc.0.l += color.0.l;
            acc.0.a += color.0.a;
            acc.0.b += color.0.b;
            acc.1 += color.1;
            acc
        },
    );
    let center = center / weight;

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

    let mut cluster_assignments = vec![0; candidates.len()];
    candidates
        .par_iter()
        .copied()
        .zip(&mut cluster_assignments)
        .for_each(|((color, _), slot)| {
            let nearest = centroids.nearest_one::<SquaredEuclidean>(&[color.l, color.a, color.b]);
            *slot = nearest.item;
        });

    let cluster_means = array::from_fn::<_, PALETTE_SIZE, _>(|_| {
        (
            [
                AtomicF32::new(0.0),
                AtomicF32::new(0.0),
                AtomicF32::new(0.0),
            ],
            AtomicF32::new(0.0),
        )
    });

    cluster_assignments
        .par_iter()
        .enumerate()
        .for_each(|(idx, &slot)| {
            cluster_means[slot as usize].0[0]
                .fetch_add(candidates[idx].0.l * candidates[idx].1, Ordering::Relaxed);
            cluster_means[slot as usize].0[1]
                .fetch_add(candidates[idx].0.a * candidates[idx].1, Ordering::Relaxed);
            cluster_means[slot as usize].0[2]
                .fetch_add(candidates[idx].0.b * candidates[idx].1, Ordering::Relaxed);
            cluster_means[slot as usize]
                .1
                .fetch_add(candidates[idx].1, Ordering::Relaxed);
        });

    for _ in 0..100 {
        centroids = KdTree::<_, _, 3, 257, u32>::with_capacity(PALETTE_SIZE);

        for (cidx, (mean, count)) in cluster_means.iter().enumerate() {
            centroids.add(
                &[
                    mean[0].load(Ordering::Relaxed) / count.load(Ordering::Relaxed),
                    mean[1].load(Ordering::Relaxed) / count.load(Ordering::Relaxed),
                    mean[2].load(Ordering::Relaxed) / count.load(Ordering::Relaxed),
                ],
                cidx as u32,
            );
        }

        let shifts = candidates
            .par_iter()
            .copied()
            .zip(&mut cluster_assignments)
            .map(|((color, w), slot)| {
                let nearest =
                    centroids.nearest_one::<SquaredEuclidean>(&[color.l, color.a, color.b]);
                if *slot == nearest.item {
                    return false;
                }

                let old_slot = *slot;
                *slot = nearest.item;

                cluster_means[old_slot as usize].0[0].fetch_sub(color.l * w, Ordering::Relaxed);
                cluster_means[old_slot as usize].0[1].fetch_sub(color.a * w, Ordering::Relaxed);
                cluster_means[old_slot as usize].0[2].fetch_sub(color.b * w, Ordering::Relaxed);
                cluster_means[old_slot as usize]
                    .1
                    .fetch_sub(w, Ordering::Relaxed);

                cluster_means[nearest.item as usize].0[0].fetch_add(color.l * w, Ordering::Relaxed);
                cluster_means[nearest.item as usize].0[1].fetch_add(color.a * w, Ordering::Relaxed);
                cluster_means[nearest.item as usize].0[2].fetch_add(color.b * w, Ordering::Relaxed);
                cluster_means[nearest.item as usize]
                    .1
                    .fetch_add(w, Ordering::Relaxed);

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
