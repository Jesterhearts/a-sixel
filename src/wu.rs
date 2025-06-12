//! Uses Wu's quantization algorithm to build a palette from an image.
//!
//! This algorithm uses principal component analysis (PCA) to recursively divide
//! the color space along the axis of greatest variance, until the
//! desired palette size is reached.

use std::{
    cmp::Ordering,
    collections::{
        BinaryHeap,
        HashSet,
    },
};

use ndarray::Array2;
use ordered_float::OrderedFloat;
use palette::{
    color_difference::EuclideanDistance,
    Lab,
};
use rayon::{
    iter::{
        IntoParallelRefIterator,
        ParallelIterator,
    },
    slice::ParallelSliceMut,
};
use rustyml::utility::principal_component_analysis::PCA;

use crate::{
    dither::Sierra,
    private,
    rgb_to_lab,
    PaletteBuilder,
    SixelEncoder,
};

pub type WuSixelEncoderMono<D = Sierra> = SixelEncoder<WuPaletteBuilder<2>, D>;
pub type WuSixelEncoder4<D = Sierra> = SixelEncoder<WuPaletteBuilder<4>, D>;
pub type WuSixelEncoder8<D = Sierra> = SixelEncoder<WuPaletteBuilder<8>, D>;
pub type WuSixelEncoder16<D = Sierra> = SixelEncoder<WuPaletteBuilder<16>, D>;
pub type WuSixelEncoder32<D = Sierra> = SixelEncoder<WuPaletteBuilder<32>, D>;
pub type WuSixelEncoder64<D = Sierra> = SixelEncoder<WuPaletteBuilder<64>, D>;
pub type WuSixelEncoder128<D = Sierra> = SixelEncoder<WuPaletteBuilder<128>, D>;
pub type WuSixelEncoder256<D = Sierra> = SixelEncoder<WuPaletteBuilder<256>, D>;

#[derive(Debug)]
struct Hist {
    points: Vec<Lab>,
    mean: Lab,
    variance: OrderedFloat<f32>,
}

impl Hist {
    fn new(points: Vec<Lab>) -> Self {
        let count = points.len() as f32;
        let sum = points
            .par_iter()
            .copied()
            .reduce(|| <Lab>::new(0.0, 0.0, 0.0), |acc, p| acc + p);
        let mean = sum / count;

        let variance = points
            .par_iter()
            .map(|p| p.distance_squared(mean))
            .sum::<f32>()
            / count;

        Self {
            points,
            mean,
            variance: OrderedFloat(variance),
        }
    }

    fn split(&mut self) -> (Self, Self) {
        let data = Array2::from_shape_fn((self.points.len(), 3), |(i, j)| match j {
            0 => self.points[i].l as f64,
            1 => self.points[i].a as f64,
            2 => self.points[i].b as f64,
            _ => unreachable!(),
        });

        let mut pca = PCA::new(3);

        match pca.fit_transform(data.view()) {
            Ok(projection) => {
                let mut projections = projection
                    .column(0)
                    .into_iter()
                    .zip(self.points.iter())
                    .map(|(proj, point)| (*proj as f32, *point))
                    .collect::<Vec<_>>();
                projections.par_sort_by_key(|(v, _)| OrderedFloat(*v));

                let left = projections[..projections.len() / 2]
                    .iter()
                    .copied()
                    .map(|(_, p)| p)
                    .collect::<Vec<_>>();
                let right = projections[projections.len() / 2..]
                    .iter()
                    .copied()
                    .map(|(_, p)| p)
                    .collect::<Vec<_>>();

                (Self::new(left), Self::new(right))
            }
            Err(_) => {
                let l_var = self
                    .points
                    .par_iter()
                    .map(|p| (p.l - self.mean.l).powi(2))
                    .sum::<f32>();

                let a_var = self
                    .points
                    .par_iter()
                    .map(|p| (p.a - self.mean.a).powi(2))
                    .sum::<f32>();

                let b_var = self
                    .points
                    .par_iter()
                    .map(|p| (p.b - self.mean.b).powi(2))
                    .sum::<f32>();

                if l_var >= a_var && l_var >= b_var {
                    self.points.sort_by_key(|p| OrderedFloat(p.l));
                } else if a_var >= b_var {
                    self.points.sort_by_key(|p| OrderedFloat(p.a));
                } else {
                    self.points.sort_by_key(|p| OrderedFloat(p.b));
                }

                let left_points = self.points[..self.points.len() / 2].to_vec();
                let right_points = self.points[self.points.len() / 2..].to_vec();

                (Self::new(left_points), Self::new(right_points))
            }
        }
    }
}

impl PartialEq for Hist {
    fn eq(&self, other: &Self) -> bool {
        self.variance == other.variance
    }
}

impl Eq for Hist {}

impl PartialOrd for Hist {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hist {
    fn cmp(&self, other: &Self) -> Ordering {
        self.variance.cmp(&other.variance)
    }
}

pub struct WuPaletteBuilder<const PALETTE_SIZE: usize>;

impl<const PALETTE_SIZE: usize> private::Sealed for WuPaletteBuilder<PALETTE_SIZE> {}
impl<const PALETTE_SIZE: usize> PaletteBuilder for WuPaletteBuilder<PALETTE_SIZE> {
    const NAME: &'static str = "Wu";
    const PALETTE_SIZE: usize = PALETTE_SIZE;

    fn build_palette(image: &image::RgbImage) -> Vec<Lab> {
        let lab_points: Vec<Lab> = image.pixels().copied().map(rgb_to_lab).collect();

        let mut heap = BinaryHeap::new();
        heap.push(Hist::new(lab_points));

        while heap.len() < PALETTE_SIZE {
            let Some(mut hist) = heap.pop() else {
                break;
            };

            let (left, right) = hist.split();
            if !left.points.is_empty() {
                heap.push(left);
            }
            if !right.points.is_empty() {
                heap.push(right);
            }
        }

        heap.into_iter()
            .map(|hist| {
                [
                    OrderedFloat(hist.mean.l),
                    OrderedFloat(hist.mean.a),
                    OrderedFloat(hist.mean.b),
                ]
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .map(|[l, a, b]| Lab::new(*l, *a, *b))
            .collect()
    }
}
