//! Use Adaptive Distributive Units to "learn" the image's color properties and
//! select palette entries.
//!
//! See <httpe://faculty.uca.edu/ecelebi/documents/ISJ_2014.pdf> for the
//! original paper on this algorithm.
//!
//! The parameters from the paper for a 256 color palette are:
//! - THETA = (400 * 256^0.5) = 6400
//! - STEPS = (2 * 256 - 3) * THETA = 3257600
//! - GAMMA = 0.015 or GAMMA_DIV ~= 64

use std::collections::HashSet;

use image::RgbImage;
use kiddo::{
    SquaredEuclidean,
    float::kdtree::KdTree,
};
use ordered_float::OrderedFloat;
use palette::Lab;
use rayon::iter::{
    IntoParallelRefIterator,
    ParallelIterator,
};
use sobol_burley::sample_4d;

use crate::{
    PaletteBuilder,
    private,
    rgb_to_lab,
};

pub struct ADUPaletteBuilder;

impl private::Sealed for ADUPaletteBuilder {}
impl PaletteBuilder for ADUPaletteBuilder {
    const NAME: &'static str = "ADU";

    fn build_palette(image: &RgbImage, palette_size: usize) -> Vec<Lab> {
        let theta = (400.0 * (palette_size as f32).sqrt()) as usize;
        let steps = ((2 * palette_size).max(4) - 3) * theta;

        let gamma: f32 = 1.0 / 64.0;

        let candidates = image.pixels().copied().map(rgb_to_lab).collect::<Vec<_>>();

        let centroid = candidates.par_iter().copied().reduce(
            || <Lab>::new(0.0, 0.0, 0.0),
            |mut acc, color| {
                acc.l += color.l;
                acc.a += color.a;
                acc.b += color.b;
                acc
            },
        ) / candidates.len() as f32;

        let mut palette = vec![centroid; palette_size];

        let mut tree = KdTree::<_, _, 3, 257, u32>::with_capacity(palette_size);
        tree.add(&[palette[0].l, palette[0].a, palette[0].b], 0);

        let mut next_idx = 1;

        let mut wc = vec![0; palette_size];

        let candidates = (0..steps as u32 / 4)
            .flat_map(|idx| {
                let [x, y, z, w] = sample_4d(idx % (1 << 16), 0, idx / (1 << 16));
                [
                    candidates[(x * candidates.len() as f32) as usize],
                    candidates[(y * candidates.len() as f32) as usize],
                    candidates[(z * candidates.len() as f32) as usize],
                    candidates[(w * candidates.len() as f32) as usize],
                ]
            })
            .collect::<Vec<_>>();

        for candidate in candidates {
            let winner =
                tree.nearest_one::<SquaredEuclidean>(&[candidate.l, candidate.a, candidate.b]);

            tree.remove(
                &[
                    palette[winner.item].l,
                    palette[winner.item].a,
                    palette[winner.item].b,
                ],
                winner.item,
            );

            palette[winner.item].l += (candidate.l - palette[winner.item].l) * gamma;
            palette[winner.item].a += (candidate.a - palette[winner.item].a) * gamma;
            palette[winner.item].b += (candidate.b - palette[winner.item].b) * gamma;

            tree.add(
                &[
                    palette[winner.item].l,
                    palette[winner.item].a,
                    palette[winner.item].b,
                ],
                winner.item,
            );

            wc[winner.item] += 1;

            if wc[winner.item] >= theta && next_idx < palette_size {
                tree.add(&[candidate.l, candidate.a, candidate.b], next_idx);

                wc[winner.item] = 0;
                wc[next_idx] = 0;
                next_idx += 1;
            }
        }

        palette
            .into_iter()
            .map(|lab| {
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
