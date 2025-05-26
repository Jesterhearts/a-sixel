//! Use Adaptive Distributive Units to "learn" the image's color properties and
//! select palette entries.
//!
//! See <httpe://faculty.uca.edu/ecelebi/documents/ISJ_2014.pdf> for the
//! original paper on this algorithm. This does slightly different parameters
//! than the paper, but the algorithm is the same core idea and these parameters
//! work well enough. See the code for the type aliases (e.g.
//! [`ADUSixelEncoder`](crate::ADUSixelEncoder)) for more default parameters.
//!
//! The parameters from the paper for a 256 color palette are:
//! - THETA = (400 * 256^0.5) = 6400
//! - STEPS = (2 * 256 - 3) * THETA = 3257600
//! - GAMMA = 0.015 or GAMMA_DIV ~= 64
//!
//! as is specified by the default arguments to this struct.

use std::collections::HashSet;

use image::RgbImage;
use kiddo::{
    float::kdtree::KdTree,
    SquaredEuclidean,
};
use ordered_float::OrderedFloat;
use palette::Lab;
use rayon::iter::{
    IntoParallelRefIterator,
    ParallelIterator,
};
use sobol_burley::sample_4d;

use crate::{
    dither::Sierra,
    private,
    rgb_to_lab,
    PaletteBuilder,
    SixelEncoder,
};

pub type ADUSixelEncoderMono<D = Sierra> = SixelEncoder<ADUPaletteBuilder<2, 400, 400>, D>;
pub type ADUSixelEncoder4<D = Sierra> = SixelEncoder<ADUPaletteBuilder<4, 800, 4000>, D>;
pub type ADUSixelEncoder8<D = Sierra> = SixelEncoder<ADUPaletteBuilder<8, 1131, 14703>, D>;
pub type ADUSixelEncoder16<D = Sierra> = SixelEncoder<ADUPaletteBuilder<16, 1600, 46400>, D>;
pub type ADUSixelEncoder32<D = Sierra> = SixelEncoder<ADUPaletteBuilder<32, 2262, 137982>, D>;
pub type ADUSixelEncoder64<D = Sierra> = SixelEncoder<ADUPaletteBuilder<64, 3200, 400000>, D>;
pub type ADUSixelEncoder128<D = Sierra> = SixelEncoder<ADUPaletteBuilder<128, 4525, 1144947>, D>;
pub type ADUSixelEncoder256<D = Sierra> = SixelEncoder<ADUPaletteBuilder<256>, D>;

pub struct ADUPaletteBuilder<
    const PALETTE_SIZE: usize = 256,
    const THETA: usize = 6400,
    const STEPS: usize = 3257600,
    const GAMMA_DIV: usize = 64,
>;

impl<const PALETTE_SIZE: usize, const THETA: usize, const STEPS: usize, const GAMMA_DIV: usize>
    private::Sealed for ADUPaletteBuilder<PALETTE_SIZE, THETA, STEPS, GAMMA_DIV>
{
}
impl<const PALETTE_SIZE: usize, const THETA: usize, const STEPS: usize, const GAMMA_DIV: usize>
    PaletteBuilder for ADUPaletteBuilder<PALETTE_SIZE, THETA, STEPS, GAMMA_DIV>
{
    const PALETTE_SIZE: usize = PALETTE_SIZE;

    fn build_palette(image: &RgbImage) -> Vec<Lab> {
        let gamma: f32 = 1.0 / (GAMMA_DIV as f32);

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

        let mut palette = [centroid; PALETTE_SIZE];

        let mut tree = KdTree::<_, _, 3, PALETTE_SIZE, u32>::with_capacity(PALETTE_SIZE);
        tree.add(&[palette[0].l, palette[0].a, palette[0].b], 0);

        let mut next_idx = 1;

        let mut wc = [0; PALETTE_SIZE];

        let candidates = (0..STEPS as u32 / 4)
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

            if wc[winner.item] >= THETA && next_idx < PALETTE_SIZE {
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
