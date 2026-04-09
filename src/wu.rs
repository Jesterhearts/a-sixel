//! Uses Wu's quantization algorithm to build a palette from an image.
//!
//! This algorithm uses principal component analysis (PCA) to recursively divide
//! the color space along the axis of greatest variance, until the
//! desired palette size is reached.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashSet;

use nalgebra_sparse::CooMatrix;
use nalgebra_sparse::CsrMatrix;
use ordered_float::OrderedFloat;
use palette::Lab;
use palette::color_difference::EuclideanDistance;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSliceMut;
use single_algebra::dimred::pca::SparsePCABuilder;

use crate::rgba_to_lab;

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
        let Ok(data) = CooMatrix::try_from_triplets_iter(
            self.points.len(),
            3,
            (0..self.points.len()).flat_map(|i| {
                [
                    (i, 1, self.points[i].l as f32),
                    (i, 2, self.points[i].a as f32),
                    (i, 3, self.points[i].b as f32),
                ]
                .into_iter()
            }),
        ) else {
            return self.split_fallback();
        };

        let data = CsrMatrix::from(&data);

        let mut pca = SparsePCABuilder::new().build();
        let Ok(projection) = pca.fit_transform(&data) else {
            return self.split_fallback();
        };

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

    fn split_fallback(&mut self) -> (Self, Self) {
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

impl PartialEq for Hist {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.variance == other.variance
    }
}

impl Eq for Hist {}

impl PartialOrd for Hist {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hist {
    fn cmp(
        &self,
        other: &Self,
    ) -> Ordering {
        self.variance.cmp(&other.variance)
    }
}

/// Builds a palette using Wu's PCA-based quantization.
///
/// Recursively bisects the color population along its principal component
/// (axis of greatest variance) until the desired number of palette entries
/// is reached.
pub struct WuPaletteBuilder;

impl WuPaletteBuilder {
    /// Quantize the image into `palette_size` colors using PCA-based
    /// bisection and return the resulting palette in Lab color space.
    pub fn build_palette(
        image: &image::RgbaImage,
        palette_size: usize,
    ) -> Vec<Lab> {
        let lab_points: Vec<Lab> = image.pixels().copied().map(rgba_to_lab).collect();

        let mut heap = BinaryHeap::new();
        heap.push(Hist::new(lab_points));

        while heap.len() < palette_size {
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
