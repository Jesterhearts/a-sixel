use image::RgbImage;
use ordered_float::NotNan;
use palette::Lab;
use rayon::{
    iter::{
        IndexedParallelIterator,
        IntoParallelRefIterator,
        IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::ParallelSliceMut,
};

use crate::{
    PaletteBuilder,
    private,
    rgb_to_lab,
};

/// https://en.wikipedia.org/wiki/Median_cut
///
/// Does a simple median cut over the input pixels.
/// - Find the bucket of pixels with the largest range on one of the three axes
///   (L, a, b).
/// - Sort the pixels in that bucket by that axis.
/// - Split the bucket in half at the median and create a new bucket for the
///   second half.
/// - Repeat until the desired number of buckets is reached.
///
/// The resulting palette is the average color of each bucket.
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
            let (best_bucket, max_idx, _) = buckets
                .par_iter()
                .zip(bucket_stats.par_iter_mut())
                .enumerate()
                .map(|(idx, (candidates, stats))| {
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

                    (idx, max_range_idx, range[max_range_idx])
                })
                .reduce(
                    || (0, 0, 0.0),
                    |a, b| {
                        if a.2 > b.2 { a } else { b }
                    },
                );

            let candidates = &mut buckets[best_bucket];
            candidates.par_sort_by(|a, b| match max_idx {
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
