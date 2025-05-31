//! Use k-means clustering to determine a palette for the image.

use std::{
    array,
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
    IntoColor,
    Lab,
    Srgb,
};
use rayon::iter::{
    IndexedParallelIterator,
    IntoParallelRefIterator,
    ParallelIterator,
};

use crate::{
    dither::Sierra,
    private,
    rgb_to_lab,
    BitPaletteBuilder,
    PaletteBuilder,
    SixelEncoder,
};

pub type KMeansSixelEncoderMono<D = Sierra> = SixelEncoder<KMeansPaletteBuilder<2>, D>;
pub type KMeansSixelEncoder4<D = Sierra> = SixelEncoder<KMeansPaletteBuilder<4>, D>;
pub type KMeansSixelEncoder8<D = Sierra> = SixelEncoder<KMeansPaletteBuilder<8>, D>;
pub type KMeansSixelEncoder16<D = Sierra> = SixelEncoder<KMeansPaletteBuilder<16>, D>;
pub type KMeansSixelEncoder32<D = Sierra> = SixelEncoder<KMeansPaletteBuilder<32>, D>;
pub type KMeansSixelEncoder64<D = Sierra> = SixelEncoder<KMeansPaletteBuilder<64>, D>;
pub type KMeansSixelEncoder128<D = Sierra> = SixelEncoder<KMeansPaletteBuilder<128>, D>;
pub type KMeansSixelEncoder256<D = Sierra> = SixelEncoder<KMeansPaletteBuilder<256>, D>;

/// Performs K-Means clustering on the image's pixels to build a palette.
pub struct KMeansPaletteBuilder<const PALETTE_SIZE: usize>;

impl<const PALETTE_SIZE: usize> private::Sealed for KMeansPaletteBuilder<PALETTE_SIZE> {}

impl<const PALETTE_SIZE: usize> PaletteBuilder for KMeansPaletteBuilder<PALETTE_SIZE> {
    const NAME: &'static str = "K-Means";
    const PALETTE_SIZE: usize = PALETTE_SIZE;

    fn build_palette(image: &RgbImage) -> Vec<Lab> {
        let candidates = image
            .pixels()
            .copied()
            .map(rgb_to_lab)
            .map(|c| (c, 1.0))
            .collect::<Vec<_>>();

        parallel_kmeans::<PALETTE_SIZE>(&candidates).0
    }
}

pub(crate) fn parallel_kmeans<const PALETTE_SIZE: usize>(
    candidates: &[(Lab, f32)],
) -> (Vec<Lab>, Vec<f32>) {
    let mut centroids = KdTree::<_, _, 3, 1025, u32>::with_capacity(PALETTE_SIZE);

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
    candidates.par_iter().copied().for_each(|(color, w)| {
        let color: Srgb = color.into_color();
        let index = BitPaletteBuilder::<BUCKETS>::index(color.into_format());
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

    for idx in 1..PALETTE_SIZE {
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
        let rgb = Srgb::new(
            maximin.0[0].load(Ordering::Relaxed) / count,
            maximin.0[1].load(Ordering::Relaxed) / count,
            maximin.0[2].load(Ordering::Relaxed) / count,
        );
        let mean: Lab = rgb.into_color();
        centroids.add(&[mean.l, mean.a, mean.b], idx as u32);
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
        centroids = KdTree::<_, _, 3, 1025, u32>::with_capacity(PALETTE_SIZE);

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

    cluster_means
        .iter()
        .map(|(mean, count)| {
            let l = mean[0].load(Ordering::Relaxed) / count.load(Ordering::Relaxed);
            let a = mean[1].load(Ordering::Relaxed) / count.load(Ordering::Relaxed);
            let b = mean[2].load(Ordering::Relaxed) / count.load(Ordering::Relaxed);
            (Lab::new(l, a, b), count.load(Ordering::Relaxed))
        })
        .unzip()
}
