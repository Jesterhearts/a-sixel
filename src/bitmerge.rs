//! Uses `BitSixelEncoder` with k-means and agglomerative merging to build a
//! palette.
//!
//! This encoder offers the best tradeoffs between speed and quality. You can
//! customize the parameters to produce speed nearly as good as
//! `BitSixelEncoder`, while producing superior results, or you can produce
//! results as good as or better than `KMeansSixelEncoder`, while being much
//! faster.
//!
//! The default parameters are tuned to produce results similar to k-means,
//! while being ~5x faster.
//!
//! # Scaling:
//! - `STAGE_1_PALETTE_SIZE`: The target size of the palette as a result of the
//!   first bit-bucketing pass. These buckets will then be passed into k-means.
//!   Time-taken scales somewhat linearly with this value.
//! - `STAGE_2_PALETTE_SIZE`: The target size of the palette as a result of the
//!   k-means clustering. This will then go through variance-minimizing
//!   agglomerative merging to produce the final palette. Time-taken scales
//!   **quadratically** with this value.

use std::{
    cmp::Reverse,
    collections::{
        BinaryHeap,
        HashSet,
    },
    sync::atomic::Ordering,
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
    IntoParallelIterator,
    IntoParallelRefIterator,
    ParallelIterator,
};

use crate::{
    BitPaletteBuilder,
    PaletteBuilder,
    kmeans::parallel_kmeans,
    private,
};

pub struct BitMergePaletteBuilder<
    const STAGE_1_PALETTE_SIZE: usize = { 1 << 18 },
    const STAGE_2_PALETTE_SIZE: usize = 512,
>;

impl<const STAGE_1_PALETTE_SIZE: usize, const STAGE_2_PALETTE_SIZE: usize> private::Sealed
    for BitMergePaletteBuilder<STAGE_1_PALETTE_SIZE, STAGE_2_PALETTE_SIZE>
{
}

impl<const STAGE_1_PALETTE_SIZE: usize, const STAGE_2_PALETTE_SIZE: usize> PaletteBuilder
    for BitMergePaletteBuilder<STAGE_1_PALETTE_SIZE, STAGE_2_PALETTE_SIZE>
{
    const NAME: &'static str = "Bit-Merge";

    fn build_palette(image: &image::RgbImage, palette_size: usize) -> Vec<palette::Lab> {
        let bit = BitPaletteBuilder::new(STAGE_1_PALETTE_SIZE);
        image.par_pixels().for_each(|pixel| {
            bit.insert(palette::Srgb::<u8>::new(pixel[0], pixel[1], pixel[2]));
        });

        let candidates = bit
            .buckets
            .into_par_iter()
            .filter_map(|bucket| {
                if bucket.count.load(Ordering::Relaxed) > 0 {
                    let lab: Lab = Srgb::new(
                        (bucket.color.0.load(Ordering::Relaxed)
                            / bucket.count.load(Ordering::Relaxed)) as u8,
                        (bucket.color.1.load(Ordering::Relaxed)
                            / bucket.count.load(Ordering::Relaxed)) as u8,
                        (bucket.color.2.load(Ordering::Relaxed)
                            / bucket.count.load(Ordering::Relaxed)) as u8,
                    )
                    .into_format()
                    .into_color();
                    Some((lab, bucket.count.load(Ordering::Relaxed) as f32))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let (mut stage2_colors, mut stage2_counts) =
            parallel_kmeans(&candidates, STAGE_2_PALETTE_SIZE);

        agglomerative_merge::<STAGE_2_PALETTE_SIZE>(
            &mut stage2_colors,
            &mut stage2_counts,
            palette_size,
        )
    }
}

pub(crate) fn agglomerative_merge<const IN_SIZE: usize>(
    stage2_colors: &mut [Lab],
    stage2_counts: &mut [f32],
    out_size: usize,
) -> Vec<Lab> {
    let mut live_stage2_colors = stage2_colors.len();
    let mut bucket_generations = [0; IN_SIZE];
    for (idx, count) in stage2_counts.iter().enumerate() {
        if *count == 0.0 {
            live_stage2_colors -= 1;
            bucket_generations[idx] = -1;
        }
    }
    let mut pqueue = BinaryHeap::new();

    let entries = stage2_colors
        .par_iter()
        .copied()
        .enumerate()
        .filter(|(idx, _)| bucket_generations[*idx] >= 0)
        .flat_map(|(idx, b_color)| {
            let bucket_generations = &bucket_generations;
            let stage2_counts = &stage2_counts;
            stage2_colors
                .par_iter()
                .enumerate()
                .skip(idx + 1)
                .filter(move |(jdx, _)| bucket_generations[*jdx] >= 0)
                .map(move |(jdx, b2_color)| {
                    let merged_var = ((stage2_counts[idx] * stage2_counts[jdx])
                        / (stage2_counts[idx] + stage2_counts[jdx]))
                        * (b_color.distance_squared(*b2_color));

                    PQueueEntry {
                        variance: Reverse(OrderedFloat(merged_var)),
                        idx1: (idx, bucket_generations[idx]),
                        idx2: (jdx, bucket_generations[jdx]),
                    }
                })
        })
        .collect::<Vec<_>>();

    for entry in entries {
        pqueue.push(entry);
    }

    while live_stage2_colors > out_size {
        let Some(PQueueEntry {
            idx1: (idx1, gen1),
            idx2: (idx2, gen2),
            ..
        }) = pqueue.pop()
        else {
            assert!(live_stage2_colors <= out_size);
            break;
        };
        if bucket_generations[idx1] != gen1 || bucket_generations[idx2] != gen2 {
            continue;
        }

        let l = (stage2_colors[idx1].l * stage2_counts[idx1]
            + stage2_colors[idx2].l * stage2_counts[idx2])
            / (stage2_counts[idx1] + stage2_counts[idx2]);
        let a = (stage2_colors[idx1].a * stage2_counts[idx1]
            + stage2_colors[idx2].a * stage2_counts[idx2])
            / (stage2_counts[idx1] + stage2_counts[idx2]);
        let b = (stage2_colors[idx1].b * stage2_counts[idx1]
            + stage2_colors[idx2].b * stage2_counts[idx2])
            / (stage2_counts[idx1] + stage2_counts[idx2]);

        stage2_colors[idx1].l = l;
        stage2_colors[idx1].a = a;
        stage2_colors[idx1].b = b;

        stage2_counts[idx1] += stage2_counts[idx2];

        bucket_generations[idx1] += 1;
        bucket_generations[idx2] = -1;
        live_stage2_colors -= 1;

        for kdx in 0..stage2_colors.len() {
            if kdx == idx1 || bucket_generations[kdx] == -1 {
                continue;
            }

            let merged_var = ((stage2_counts[idx1] * stage2_counts[kdx])
                / (stage2_counts[idx1] + stage2_counts[kdx]))
                * (stage2_colors[idx1].distance_squared(stage2_colors[kdx]));

            pqueue.push(PQueueEntry {
                variance: Reverse(OrderedFloat(merged_var)),
                idx1: (idx1, bucket_generations[idx1]),
                idx2: (kdx, bucket_generations[kdx]),
            });
        }
    }

    stage2_colors
        .iter()
        .zip(bucket_generations)
        .filter(|(_, generation)| *generation >= 0)
        .map(|(lab, _)| {
            [
                OrderedFloat(lab.l),
                OrderedFloat(lab.a),
                OrderedFloat(lab.b),
            ]
        })
        .collect::<HashSet<_>>()
        .into_iter()
        .map(|[l, a, b]| Lab::new(*l, *a, *b))
        .collect::<Vec<_>>()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PQueueEntry {
    variance: Reverse<OrderedFloat<f32>>,
    idx1: (usize, i32),
    idx2: (usize, i32),
}

impl Ord for PQueueEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.variance.cmp(&other.variance)
    }
}

impl PartialOrd for PQueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
