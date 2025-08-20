//! Uses k-medians to build a palette of colors.

use std::{
    collections::HashSet,
    sync::atomic::{
        AtomicBool,
        Ordering,
    },
};

use atomic_float::AtomicF32;
use dashmap::DashSet;
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
    BitPaletteBuilder,
    PaletteBuilder,
    private,
    rgb_to_lab,
};

pub struct KMediansPaletteBuilder;

impl private::Sealed for KMediansPaletteBuilder {}

impl PaletteBuilder for KMediansPaletteBuilder {
    const NAME: &'static str = "K-Medians";

    fn build_palette(image: &RgbImage, palette_size: usize) -> Vec<Lab> {
        let candidates = image
            .pixels()
            .copied()
            .map(rgb_to_lab)
            .map(|l| (l, 1.0))
            .collect::<Vec<_>>();

        parallel_kmedians(&candidates, palette_size)
    }
}

pub(crate) fn parallel_kmedians(candidates: &[(Lab, f32)], palette_size: usize) -> Vec<Lab> {
    let mut centroids = KdTree::<_, _, 3, 257, u32>::with_capacity(palette_size);

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

    let init = buckets
        .iter()
        .max_by_key(|bucket| OrderedFloat(bucket.1.load(Ordering::Relaxed)))
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
        let rgb = Srgb::new(
            maximin.0[0].load(Ordering::Relaxed) / count,
            maximin.0[1].load(Ordering::Relaxed) / count,
            maximin.0[2].load(Ordering::Relaxed) / count,
        );
        let mean: Lab = rgb.into_color();
        centroids.add(&[mean.l, mean.a, mean.b], idx as u32);
    }

    let mut cluster_assignments = std::iter::repeat_with(DashSet::<KLabW>::new)
        .take(palette_size)
        .collect::<Vec<_>>();
    candidates.par_iter().copied().for_each(|(color, w)| {
        let nearest = centroids.nearest_one::<SquaredEuclidean>(&[color.l, color.a, color.b]);
        cluster_assignments[nearest.item as usize].insert((color, w).into());
    });

    let mut medians = vec![[0.0; 3]; palette_size];

    for _ in 0..100 {
        centroids = KdTree::<_, _, 3, 257, u32>::with_capacity(palette_size);

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

        for (idx, medians) in medians.iter().enumerate() {
            centroids.add(medians, idx as u32);
        }

        let old_assignments = cluster_assignments;
        cluster_assignments = std::iter::repeat_with(DashSet::<KLabW>::new)
            .take(palette_size)
            .collect::<Vec<_>>();
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
