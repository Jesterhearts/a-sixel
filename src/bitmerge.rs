//! Uses BitSixelEncoder with agglomerative merging to build a palette. This
//! is roughly as fast as BitSixelEncoder, but produces a substantially
//! higher-quality palette.

use std::{
    cmp::Reverse,
    collections::{
        BinaryHeap,
        HashSet,
    },
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
    BitPaletteBuilder,
    PaletteBuilder,
    SixelEncoder,
};

pub type BitMergeSixelEncoderMono<D = Sierra, const LARGE: usize = 1024> =
    SixelEncoder<BitMergePaletteBuilder<2, LARGE>, D>;
pub type BitMergeSixelEncoder4<D = Sierra, const LARGE: usize = 1024> =
    SixelEncoder<BitMergePaletteBuilder<4, LARGE>, D>;
pub type BitMergeSixelEncoder8<D = Sierra, const LARGE: usize = 1024> =
    SixelEncoder<BitMergePaletteBuilder<8, LARGE>, D>;
pub type BitMergeSixelEncoder16<D = Sierra, const LARGE: usize = 1024> =
    SixelEncoder<BitMergePaletteBuilder<16, LARGE>, D>;
pub type BitMergeSixelEncoder32<D = Sierra, const LARGE: usize = 1024> =
    SixelEncoder<BitMergePaletteBuilder<32, LARGE>, D>;
pub type BitMergeSixelEncoder64<D = Sierra, const LARGE: usize = 1024> =
    SixelEncoder<BitMergePaletteBuilder<64, LARGE>, D>;
pub type BitMergeSixelEncoder128<D = Sierra, const LARGE: usize = 1024> =
    SixelEncoder<BitMergePaletteBuilder<128, LARGE>, D>;
pub type BitMergeSixelEncoder256<D = Sierra, const LARGE: usize = 1024> =
    SixelEncoder<BitMergePaletteBuilder<256, LARGE>, D>;

pub struct BitMergePaletteBuilder<const TARGET_PALETTE_SIZE: usize, const LARGE_PALETTE_SIZE: usize>;

impl<const TARGET_PALETTE_SIZE: usize, const LARGE_PALETTE_SIZE: usize> private::Sealed
    for BitMergePaletteBuilder<TARGET_PALETTE_SIZE, LARGE_PALETTE_SIZE>
{
}

impl<const TARGET_PALETTE_SIZE: usize, const LARGE_PALETTE_SIZE: usize> PaletteBuilder
    for BitMergePaletteBuilder<TARGET_PALETTE_SIZE, LARGE_PALETTE_SIZE>
{
    const NAME: &'static str = "Bit-Merge";
    const PALETTE_SIZE: usize = TARGET_PALETTE_SIZE;

    fn build_palette(image: &image::RgbImage) -> Vec<palette::Lab> {
        let mut bit = BitPaletteBuilder::<LARGE_PALETTE_SIZE>::new();
        for pixel in image.pixels() {
            bit.insert(palette::Srgb::<u8>::new(pixel[0], pixel[1], pixel[2]));
        }

        let mut buckets = bit.buckets;
        let mut live_buckets = buckets.len();
        let mut bucket_generations = [0; LARGE_PALETTE_SIZE];
        for (idx, bucket) in buckets.iter().enumerate() {
            if bucket.count == 0 {
                live_buckets -= 1;
                bucket_generations[idx] = -1;
            }
        }
        let mut pqueue = BinaryHeap::new();

        let entries = buckets
            .par_iter()
            .copied()
            .enumerate()
            .filter(|(_, bucket)| bucket.count > 0)
            .flat_map(|(idx, bucket)| {
                let b_color: Lab = Srgb::new(
                    (bucket.color.0 / bucket.count) as u8,
                    (bucket.color.1 / bucket.count) as u8,
                    (bucket.color.2 / bucket.count) as u8,
                )
                .into_format()
                .into_color();
                let bucket_generations = &bucket_generations;
                buckets
                    .par_iter()
                    .enumerate()
                    .skip(idx + 1)
                    .filter(move |(_, bucket2)| bucket2.count > 0)
                    .map(move |(jdx, bucket2)| {
                        let b2_color: Lab = Srgb::new(
                            (bucket2.color.0 / bucket2.count) as u8,
                            (bucket2.color.1 / bucket2.count) as u8,
                            (bucket2.color.2 / bucket2.count) as u8,
                        )
                        .into_format()
                        .into_color();

                        let merged_var = ((bucket.count * bucket2.count) as f32
                            / (bucket.count + bucket2.count) as f32)
                            * (b_color.distance_squared(b2_color));

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

        while live_buckets > TARGET_PALETTE_SIZE {
            let Some(PQueueEntry {
                idx1: (idx1, gen1),
                idx2: (idx2, gen2),
                ..
            }) = pqueue.pop()
            else {
                assert!(live_buckets <= TARGET_PALETTE_SIZE);
                break;
            };
            if bucket_generations[idx1] != gen1 || bucket_generations[idx2] != gen2 {
                continue;
            }

            buckets[idx1].color.0 += buckets[idx2].color.0;
            buckets[idx1].color.1 += buckets[idx2].color.1;
            buckets[idx1].color.2 += buckets[idx2].color.2;
            buckets[idx1].count += buckets[idx2].count;
            bucket_generations[idx1] += 1;
            bucket_generations[idx2] = -1;
            live_buckets -= 1;

            let b_color: Lab = Srgb::new(
                (buckets[idx1].color.0 / buckets[idx1].count) as u8,
                (buckets[idx1].color.1 / buckets[idx1].count) as u8,
                (buckets[idx1].color.2 / buckets[idx1].count) as u8,
            )
            .into_format()
            .into_color();

            for (kdx, bucket) in buckets.iter().enumerate() {
                if kdx == idx1 || bucket_generations[kdx] == -1 {
                    continue;
                }

                let b2_color: Lab = Srgb::new(
                    (bucket.color.0 / bucket.count) as u8,
                    (bucket.color.1 / bucket.count) as u8,
                    (bucket.color.2 / bucket.count) as u8,
                )
                .into_format()
                .into_color();

                let merged_var = ((buckets[idx1].count * bucket.count) as f32
                    / (buckets[idx1].count + bucket.count) as f32)
                    * (b_color.distance_squared(b2_color));

                pqueue.push(PQueueEntry {
                    variance: Reverse(OrderedFloat(merged_var)),
                    idx1: (idx1, bucket_generations[idx1]),
                    idx2: (kdx, bucket_generations[kdx]),
                });
            }
        }

        buckets
            .into_iter()
            .zip(bucket_generations)
            .filter(|(_, generation)| *generation >= 0)
            .map(|(node, _)| {
                let rgb = Srgb::new(
                    (node.color.0 / node.count) as u8,
                    (node.color.1 / node.count) as u8,
                    (node.color.2 / node.count) as u8,
                );
                let lab: Lab = rgb.into_format().into_color();
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
