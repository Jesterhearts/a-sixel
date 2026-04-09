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

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashSet;

use ordered_float::OrderedFloat;
use palette::IntoColor;
use palette::Lab;
use palette::Srgb;
use palette::color_difference::EuclideanDistance;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

use crate::bit::BitPaletteBuilder;
use crate::kmeans::parallel_kmeans;

/// Builds a palette by combining bit-dilation bucketing, k-means refinement,
/// and variance-minimizing agglomerative merging.
///
/// The const generic parameters control the two-stage pipeline:
/// - `STAGE_1_PALETTE_SIZE` is the number of bit-buckets fed into k-means
///   (scales roughly linearly with time).
/// - `STAGE_2_PALETTE_SIZE` is the k-means output size that feeds into the
///   agglomerative merge step (scales **quadratically** with time).
///
/// The default parameters target quality similar to
/// [`KMeansPaletteBuilder`](crate::kmeans::KMeansPaletteBuilder) at roughly 5x
/// the speed.
pub struct BitMergePaletteBuilder<
    const STAGE_1_PALETTE_SIZE: usize = { 1 << 18 },
    const STAGE_2_PALETTE_SIZE: usize = 512,
>;

impl<const STAGE_1_PALETTE_SIZE: usize, const STAGE_2_PALETTE_SIZE: usize>
    BitMergePaletteBuilder<STAGE_1_PALETTE_SIZE, STAGE_2_PALETTE_SIZE>
{
    /// Quantize the image into `palette_size` colors using the bit-merge
    /// pipeline and return the resulting palette in Lab color space.
    pub fn build_palette(
        image: &image::RgbaImage,
        palette_size: usize,
    ) -> Vec<palette::Lab> {
        let bit = BitPaletteBuilder::new(STAGE_1_PALETTE_SIZE);

        thread_local! {
            static PALETTE: RefCell<Vec<(u64, u64, u64, u64)>> = RefCell::default();
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(rayon::current_num_threads())
            .build()
            .unwrap();

        pool.install(|| {
            image.par_pixels().for_each(|pixel| {
                PALETTE.with_borrow_mut(|palette| {
                    palette.resize(STAGE_1_PALETTE_SIZE, (0, 0, 0, 0));

                    let pixel = Srgb::<u8>::new(pixel[0], pixel[1], pixel[2]);
                    let index = BitPaletteBuilder::index(pixel, bit.shift);
                    palette[index].0 += pixel.red as u64;
                    palette[index].1 += pixel.green as u64;
                    palette[index].2 += pixel.blue as u64;
                    palette[index].3 += 1;
                });
            });
        });

        let per_thread_palettes = pool.broadcast(|_ctx| PALETTE.with_borrow_mut(std::mem::take));

        let mut final_palette = vec![(0, 0, 0, 0); STAGE_1_PALETTE_SIZE];
        for palette in per_thread_palettes {
            for (dest, src) in final_palette.iter_mut().zip(palette) {
                dest.0 += src.0;
                dest.1 += src.1;
                dest.2 += src.2;
                dest.3 += src.3;
            }
        }

        let candidates = final_palette
            .into_iter()
            .filter(|node| node.3 > 0)
            .map(|node| {
                let rgb = Srgb::new(
                    (node.0 / node.3) as u8,
                    (node.1 / node.3) as u8,
                    (node.2 / node.3) as u8,
                );
                let lab: Lab = rgb.into_format().into_color();
                (lab, node.3 as f32)
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
    fn cmp(
        &self,
        other: &Self,
    ) -> std::cmp::Ordering {
        self.variance.cmp(&other.variance)
    }
}

impl PartialOrd for PQueueEntry {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
