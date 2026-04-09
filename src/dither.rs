//! A collection of various dithering algorithms that can be used with the
//! [`SixelEncoder`](crate::SixelEncoder) to dither the result.
//!
//! Most of the dithering algorithms are based on the ones described in
//! [this article](https://tannerhelland.com/2012/12/28/dithering-eleven-algorithms-source-code.html).
//! There is also an implementation of the Bayer matrix dithering algorithm
//! and a Sobol sequence based dithering algorithm.

use dilate::DilateExpand;
use image::Rgba;
use image::RgbaImage;
use kiddo::float::kdtree::KdTree;
use palette::Lab;
use palette::color_difference::EuclideanDistance;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSliceMut;
use sobol_burley::sample;
#[cfg(feature = "strum")]
use strum::Display;
#[cfg(feature = "strum")]
use strum::EnumString;

use crate::rgba_to_lab;

/// Abstracts nearest-neighbor palette lookup, allowing different strategies
/// (e.g. KD-tree vs. direct bucket lookup) to be used interchangeably.
pub enum PaletteBucketer {
    /// Backed by a KD-tree for nearest-neighbor queries in Lab color space.
    KdTree(KdTreeBucketer),
    /// Direct bit-dilation bucketing for fast lookups without color-space
    /// conversion.
    Bit(crate::bit::BitPaletteBucketer),
}

impl PaletteBucketer {
    /// Find the nearest palette entry for a point in Lab color space.
    fn nearest(
        &self,
        point: &[f32; 3],
    ) -> usize {
        match self {
            Self::KdTree(b) => b.nearest(point),
            Self::Bit(b) => b.nearest(point),
        }
    }

    /// Find the two nearest palette entries for a point in Lab color space,
    /// returned as `(palette_index, squared_distance)` pairs ordered nearest
    /// first. Used by ordered-dithering algorithms that interpolate between
    /// the two closest colors.
    fn nearest_two(
        &self,
        point: Rgba<u8>,
    ) -> [(usize, f32); 2] {
        match self {
            Self::KdTree(b) => b.nearest_two(point),
            Self::Bit(b) => b.nearest_two(point),
        }
    }

    /// Find the nearest palette entry for an RGB pixel.
    fn nearest_rgb(
        &self,
        pixel: Rgba<u8>,
    ) -> usize {
        match self {
            Self::KdTree(b) => b.nearest_rgb(pixel),
            Self::Bit(b) => b.nearest_rgb(pixel),
        }
    }
}

/// A [`PaletteBucketer`] variant backed by a KD-tree for nearest-neighbor
/// queries in Lab color space.
pub struct KdTreeBucketer(KdTree<f32, usize, 3, 257, u32>);

impl KdTreeBucketer {
    pub fn new(palette: &[Lab]) -> Self {
        let mut tree = KdTree::with_capacity(palette.len());
        for (idx, color) in palette.iter().enumerate() {
            tree.add(color.as_ref(), idx);
        }
        Self(tree)
    }

    fn nearest(
        &self,
        point: &[f32; 3],
    ) -> usize {
        self.0.nearest_one::<kiddo::SquaredEuclidean>(point).item
    }

    fn nearest_two(
        &self,
        point: Rgba<u8>,
    ) -> [(usize, f32); 2] {
        let point = rgba_to_lab(point);
        let point = [point.l, point.a, point.b];
        let [l1, l2] = self
            .0
            .nearest_n::<kiddo::SquaredEuclidean>(&point, 2)
            .try_into()
            .unwrap();
        [(l1.item, l1.distance), (l2.item, l2.distance)]
    }

    fn nearest_rgb(
        &self,
        pixel: Rgba<u8>,
    ) -> usize {
        let lab = rgba_to_lab(pixel);
        self.nearest(lab.as_ref())
    }
}

/// The dithering algorithm to apply during sixel encoding.
///
/// # Choosing an algorithm
/// - [`Dither::Sierra`] is a good default choice for dithering, as it produces
///   high-quality results with minimal artifacts.
/// - [`Dither::None`] can be used if performance is a concern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "strum", derive(EnumString, Display))]
#[cfg_attr(
    feature = "strum",
    strum(ascii_case_insensitive, serialize_all = "kebab-case")
)]
#[non_exhaustive]
pub enum Dither {
    /// Do not perform dithering. This is the fastest possible palette encoding
    /// option.
    None,
    /// Sierra dithering (3-row).
    Sierra,
    /// Sierra-2 dithering (2-row).
    Sierra2,
    /// Sierra Lite dithering.
    SierraLite,
    /// Floyd-Steinberg dithering.
    FloydSteinberg,
    /// Jarvis, Judice, and Ninke dithering.
    JJN,
    /// Stucki dithering.
    Stucki,
    /// Atkinson dithering.
    Atkinson,
    /// Burkes dithering.
    Burkes,
    /// Bayer ordered dithering.
    Bayer,
    /// Uses the Sobol sequence to perturb the colors in a quasi-random manner.
    /// This is somewhere between ordered dithering and stochastic dithering.
    /// The results are generally decent, although they may appear textured like
    /// paper.
    Sobol,
}

impl Dither {
    /// Take the input image and convert it to the input palette, applying a
    /// dithering algorithm to the result.
    pub fn dither_and_palettize(
        &self,
        image: &RgbaImage,
        in_palette: &[Lab],
        bucketer: &PaletteBucketer,
    ) -> Vec<usize> {
        match self {
            Self::None => no_dither(image, bucketer),
            Self::Sierra => {
                error_diffusion_dither(image, in_palette, bucketer, SIERRA_KERNEL, 32.0)
            }
            Self::Sierra2 => {
                error_diffusion_dither(image, in_palette, bucketer, SIERRA2_KERNEL, 16.0)
            }
            Self::SierraLite => {
                error_diffusion_dither(image, in_palette, bucketer, SIERRA_LITE_KERNEL, 4.0)
            }
            Self::FloydSteinberg => {
                error_diffusion_dither(image, in_palette, bucketer, FLOYD_STEINBERG_KERNEL, 16.0)
            }
            Self::JJN => error_diffusion_dither(image, in_palette, bucketer, JJN_KERNEL, 48.0),
            Self::Stucki => {
                error_diffusion_dither(image, in_palette, bucketer, STUCKI_KERNEL, 42.0)
            }
            Self::Atkinson => {
                error_diffusion_dither(image, in_palette, bucketer, ATKINSON_KERNEL, 8.0)
            }
            Self::Burkes => {
                error_diffusion_dither(image, in_palette, bucketer, BURKES_KERNEL, 32.0)
            }
            Self::Bayer => bayer_dither(image, in_palette, bucketer),
            Self::Sobol => sobol_dither(image, in_palette, bucketer),
        }
    }
}

// --- Kernel constants ---

// _ _ X 5 3
// 2 4 5 4 2
// _ 2 3 2 _
const SIERRA_KERNEL: &[(isize, isize, f32)] = &[
    (1, 0, 5.0),
    (2, 0, 3.0),
    //
    (-2, 1, 2.0),
    (-1, 1, 4.0),
    (0, 1, 5.0),
    (1, 1, 4.0),
    (2, 1, 2.0),
    //
    (-1, 2, 2.0),
    (0, 2, 3.0),
    (1, 2, 2.0),
];

// _ _ X 4 3
// 1 2 3 2 1
const SIERRA2_KERNEL: &[(isize, isize, f32)] = &[
    //x y  m
    (1, 0, 4.0),
    (2, 0, 3.0),
    //
    (-2, 1, 1.0),
    (-1, 1, 2.0),
    (0, 1, 3.0),
    (1, 1, 2.0),
    (2, 1, 1.0),
];

//  _ X 2
//  1 1 _
const SIERRA_LITE_KERNEL: &[(isize, isize, f32)] = &[
    //x y  m
    (1, 0, 2.0),
    //
    (-1, 1, 1.0),
    (0, 1, 1.0),
];

// _ X 7
// 3 5 3
const FLOYD_STEINBERG_KERNEL: &[(isize, isize, f32)] = &[
    //x y  m
    (1, 0, 7.0),
    //
    (-1, 1, 3.0),
    (0, 1, 5.0),
    (1, 1, 1.0),
];

// _ _ X 7 5
// 3 5 7 5 3
// 1 2 5 3 1
const JJN_KERNEL: &[(isize, isize, f32)] = &[
    //x y  m
    (1, 0, 7.0),
    (2, 0, 5.0),
    //
    (-2, 1, 3.0),
    (-1, 1, 5.0),
    (0, 1, 7.0),
    (1, 1, 5.0),
    (2, 1, 3.0),
    //
    (-2, 2, 1.0),
    (-1, 2, 2.0),
    (0, 2, 5.0),
    (1, 2, 3.0),
    (2, 2, 1.0),
];

// _ _ X 8 4
// 2 4 8 4 2
// 1 2 4 2 1
const STUCKI_KERNEL: &[(isize, isize, f32)] = &[
    //x y  m
    (1, 0, 8.0),
    (2, 0, 4.0),
    //
    (-2, 1, 2.0),
    (-1, 1, 4.0),
    (0, 1, 8.0),
    (1, 1, 4.0),
    (2, 1, 2.0),
    //
    (-2, 2, 1.0),
    (-1, 2, 2.0),
    (0, 2, 4.0),
    (1, 2, 2.0),
    (2, 2, 1.0),
];

// _ X 1 1
// 1 1 1 _
// _ 1 _ _
const ATKINSON_KERNEL: &[(isize, isize, f32)] = &[
    //x y  m
    (1, 0, 1.0),
    (2, 0, 1.0),
    //
    (-1, 1, 1.0),
    (0, 1, 1.0),
    (1, 1, 1.0),
    //
    (0, 2, 1.0),
];

// _ _ X 8 4
// 2 4 8 4 2
const BURKES_KERNEL: &[(isize, isize, f32)] = &[
    //x y  m
    (1, 0, 8.0),
    (2, 0, 4.0),
    //
    (-2, 1, 2.0),
    (-1, 1, 4.0),
    (0, 1, 8.0),
    (1, 1, 4.0),
    (2, 1, 2.0),
];

// --- Dither implementations ---

fn no_dither(
    image: &RgbaImage,
    bucketer: &PaletteBucketer,
) -> Vec<usize> {
    let mut result = vec![0; image.width() as usize * image.height() as usize];
    image.par_pixels().zip(&mut result).for_each(|(p, dest)| {
        *dest = bucketer.nearest_rgb(*p);
    });

    result
}

fn error_diffusion_dither(
    image: &RgbaImage,
    in_palette: &[Lab],
    bucketer: &PaletteBucketer,
    kernel: &[(isize, isize, f32)],
    div: f32,
) -> Vec<usize> {
    let pixels = image.pixels().copied().map(rgba_to_lab).collect::<Vec<_>>();

    let mut result = vec![0; image.width() as usize * image.height() as usize];

    let mut spills = vec![[0.0; 3]; image.width() as usize * image.height() as usize];

    for (idx, p) in pixels.iter().copied().enumerate() {
        let pixel = <Lab>::new(
            p.l + spills[idx][0],
            p.a + spills[idx][1],
            p.b + spills[idx][2],
        );

        let palette_idx = bucketer.nearest(&[pixel.l, pixel.a, pixel.b]);

        let error0 = pixel.l - in_palette[palette_idx].l;
        let error1 = pixel.a - in_palette[palette_idx].a;
        let error2 = pixel.b - in_palette[palette_idx].b;
        let spill = [error0, error1, error2];

        result[idx] = palette_idx;

        for (dx, dy, m) in kernel {
            let x = idx as isize % image.width() as isize + dx;
            let y = idx as isize / image.width() as isize + dy;
            if x < 0 || y < 0 || x >= image.width() as isize || y >= image.height() as isize {
                continue;
            }

            let jdx = y * image.width() as isize + x;
            let target = &mut spills[jdx as usize];
            *target = [
                target[0] + (spill[0] * m) / div / 2.0,
                target[1] + (spill[1] * m) / div / 2.0,
                target[2] + (spill[2] * m) / div / 2.0,
            ];
        }
    }

    result
}

fn bayer_dither(
    image: &RgbaImage,
    in_palette: &[Lab],
    bucketer: &PaletteBucketer,
) -> Vec<usize> {
    let order = image.width().min(image.height()).ilog2().max(2) as usize;
    let matrix_size = 1 << order;
    let total_bits = 2 * order;
    let mut matrix = vec![0.0; matrix_size * matrix_size];

    (0..matrix_size as u32)
        .into_par_iter()
        .zip(matrix.par_chunks_mut(matrix_size))
        .for_each(|(y, row)| {
            for (x, cell) in row.iter_mut().enumerate() {
                let x = x as u32;
                let bits =
                    (x ^ y).dilate_expand::<2>().value() | (y.dilate_expand::<2>().value() << 1);
                let bits = bits.reverse_bits() >> (u32::BITS - total_bits as u32);
                *cell = bits as f32 / matrix_size.pow(2) as f32;
            }
        });

    let max_matrix = matrix
        .iter()
        .copied()
        .max_by(|l, r| l.total_cmp(r))
        .unwrap_or(1.0);
    let min_matrix = matrix
        .iter()
        .copied()
        .min_by(|l, r| l.total_cmp(r))
        .unwrap_or(0.0);
    matrix.par_iter_mut().for_each(|cell| {
        *cell = (*cell - min_matrix) / (max_matrix - min_matrix);
    });

    let mut result = vec![0; image.width() as usize * image.height() as usize];
    image
        .par_pixels()
        .enumerate()
        .zip(&mut result)
        .for_each(|((idx, p), dest)| {
            let x = (idx % image.width() as usize) % matrix_size;
            let y = (idx / image.width() as usize) % matrix_size;

            let [(l1_item, l1_dist), (l2_item, l2_dist)] = bucketer.nearest_two(*p);

            let p_dist = in_palette[l1_item].distance_squared(in_palette[l2_item]);
            let t = ((l1_dist - l2_dist + p_dist) / (2.0 * p_dist)).clamp(0.0, 1.0);

            let m_idx = x + y * matrix_size;
            if t > matrix[m_idx] {
                *dest = l2_item;
            } else {
                *dest = l1_item;
            }
        });

    result
}

fn sobol_dither(
    image: &RgbaImage,
    in_palette: &[Lab],
    bucketer: &PaletteBucketer,
) -> Vec<usize> {
    let mut result = vec![0; image.width() as usize * image.height() as usize];
    image
        .par_pixels()
        .enumerate()
        .zip(&mut result)
        .for_each(|((idx, p), dest)| {
            let thresh = sample(idx as u32 % (1 << 16), 0, idx as u32 / (1 << 16));

            let [(l1_item, l1_dist), (l2_item, l2_dist)] = bucketer.nearest_two(*p);

            let p_dist = in_palette[l1_item].distance_squared(in_palette[l2_item]);
            let t = ((l1_dist - l2_dist + p_dist) / (2.0 * p_dist)).clamp(0.0, 1.0);

            if t > thresh {
                *dest = l2_item;
            } else {
                *dest = l1_item;
            }
        });

    result
}
