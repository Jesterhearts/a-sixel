//! Use k-means clustering to determine a palette for the image.

use std::sync::atomic::Ordering;

use atomic_float::AtomicF32;
use image::RgbImage;
use kiddo::{
    SquaredEuclidean,
    float::kdtree::KdTree,
};
use ordered_float::OrderedFloat;
use palette::{
    IntoColor,
    Lab,
    Srgb,
    color_difference::EuclideanDistance,
};
use rayon::iter::{
    IndexedParallelIterator,
    IntoParallelRefIterator,
    ParallelIterator,
};

use crate::{
    BitPaletteBuilder,
    PaletteBuilder,
    private,
    rgb_to_lab,
};

/// Performs K-Means clustering on the image's pixels to build a palette.
pub struct KMeansPaletteBuilder;

impl private::Sealed for KMeansPaletteBuilder {}

impl PaletteBuilder for KMeansPaletteBuilder {
    const NAME: &'static str = "K-Means";

    fn build_palette(image: &RgbImage, palette_size: usize) -> Vec<Lab> {
        let candidates = image
            .pixels()
            .copied()
            .map(rgb_to_lab)
            .map(|c| (c, 1.0))
            .collect::<Vec<_>>();

        parallel_kmeans(&candidates, palette_size).0
    }
}

pub(crate) fn parallel_kmeans(
    candidates: &[(Lab, f32)],
    palette_size: usize,
) -> (Vec<Lab>, Vec<f32>) {
    let mut centroids = KdTree::<_, _, 3, 1025, u32>::with_capacity(palette_size);

    const BUCKETS: usize = 1 << 14;
    let buckets = Vec::from_iter(
        std::iter::repeat_with(|| {
            (
                [
                    AtomicF32::new(0.0),
                    AtomicF32::new(0.0),
                    AtomicF32::new(0.0),
                ],
                AtomicF32::new(0.0),
            )
        })
        .take(BUCKETS),
    );

    let shift = BitPaletteBuilder::shift(BUCKETS);
    candidates.par_iter().copied().for_each(|(color, w)| {
        let color: Srgb = color.into_color();
        let index = BitPaletteBuilder::index(color.into_format(), shift);
        buckets[index].0[0].fetch_add(color.red as f32 * w, Ordering::Relaxed);
        buckets[index].0[1].fetch_add(color.green as f32 * w, Ordering::Relaxed);
        buckets[index].0[2].fetch_add(color.blue as f32 * w, Ordering::Relaxed);
        buckets[index].1.fetch_add(w, Ordering::Relaxed);
    });

    let (centroid, count) = buckets
        .iter()
        .filter_map(|bucket| {
            let count = bucket.1.load(Ordering::Relaxed);
            if count > 0.0 {
                let rgb = Srgb::new(
                    bucket.0[0].load(Ordering::Relaxed),
                    bucket.0[1].load(Ordering::Relaxed),
                    bucket.0[2].load(Ordering::Relaxed),
                );
                let lab: Lab = rgb.into_color();
                Some((lab, count))
            } else {
                None
            }
        })
        .fold((<Lab>::new(0.0, 0.0, 0.0), 0.0), |mut acc, color| {
            acc.0 += color.0;
            acc.1 += color.1;
            acc
        });
    let centroid = centroid / count;

    let init = buckets
        .iter()
        .max_by_key(|bucket| {
            let count = bucket.1.load(Ordering::Relaxed);
            let rgb = Srgb::new(
                bucket.0[0].load(Ordering::Relaxed) / count,
                bucket.0[1].load(Ordering::Relaxed) / count,
                bucket.0[2].load(Ordering::Relaxed) / count,
            );
            let lab: Lab = rgb.into_color();
            let dist = centroid.distance_squared(lab);
            OrderedFloat(dist * count)
        })
        .unwrap();
    let count = init.1.load(Ordering::Relaxed);
    let rgb = Srgb::new(
        init.0[0].load(Ordering::Relaxed) / count,
        init.0[1].load(Ordering::Relaxed) / count,
        init.0[2].load(Ordering::Relaxed) / count,
    );
    let lab: Lab = rgb.into_color();
    centroids.add(&[lab.l, lab.a, lab.b], 0);

    for idx in 1..palette_size {
        let maximin = buckets
            .par_iter()
            .max_by_key(|bucket| {
                let count = bucket.1.load(Ordering::Relaxed);
                let rgb = Srgb::new(
                    bucket.0[0].load(Ordering::Relaxed) / count,
                    bucket.0[1].load(Ordering::Relaxed) / count,
                    bucket.0[2].load(Ordering::Relaxed) / count,
                );
                let lab: Lab = rgb.into_color();
                let min = centroids
                    .nearest_one::<SquaredEuclidean>(&[lab.l, lab.a, lab.b])
                    .distance;
                OrderedFloat(min * count)
            })
            .unwrap();
        let count = maximin.1.load(Ordering::Relaxed);
        if count > 0.0 {
            let rgb = Srgb::new(
                maximin.0[0].load(Ordering::Relaxed) / count,
                maximin.0[1].load(Ordering::Relaxed) / count,
                maximin.0[2].load(Ordering::Relaxed) / count,
            );
            let mean: Lab = rgb.into_color();
            centroids.add(&[mean.l, mean.a, mean.b], idx as u32);
        }
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

    let cluster_means = std::iter::repeat_with(|| {
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
        centroids = KdTree::<_, _, 3, 1025, u32>::with_capacity(palette_size);

        for (cidx, (mean, count)) in cluster_means.iter().enumerate() {
            let count = count.load(Ordering::Relaxed);
            if count > 0.0 {
                centroids.add(
                    &[
                        mean[0].load(Ordering::Relaxed) / count,
                        mean[1].load(Ordering::Relaxed) / count,
                        mean[2].load(Ordering::Relaxed) / count,
                    ],
                    cidx as u32,
                );
            }
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

    cluster_means
        .iter()
        .filter(|(_, count)| count.load(Ordering::Relaxed) > 0.0)
        .map(|(mean, count)| {
            let count = count.load(Ordering::Relaxed);
            let l = mean[0].load(Ordering::Relaxed) / count;
            let a = mean[1].load(Ordering::Relaxed) / count;
            let b = mean[2].load(Ordering::Relaxed) / count;
            (Lab::new(l, a, b), count)
        })
        .unzip()
}
