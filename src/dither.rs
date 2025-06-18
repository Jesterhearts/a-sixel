//! A collection of various dithering algorithms that can be used with the
//! [`SixelEncoder`](crate::SixelEncoder) to dither the result.
//!
//! Most of the dithering algorithms are based on the ones described in
//! [this article](https://tannerhelland.com/2012/12/28/dithering-eleven-algorithms-source-code.html).
//! There is also an implementation of the Bayer matrix dithering algorithm
//! and a Sobol sequence based dithering algorithm.

use dilate::DilateExpand;
use image::RgbImage;
use kiddo::float::kdtree::KdTree;
use palette::{
    color_difference::EuclideanDistance,
    Lab,
};
use rayon::{
    iter::{
        IndexedParallelIterator,
        IntoParallelIterator,
        ParallelIterator,
    },
    slice::ParallelSliceMut,
};
use sobol_burley::sample_4d;

use crate::{
    private,
    rgb_to_lab,
};

/// Each struct in this module implements this trait and can be combined with
/// the [`SixelEncoder`](crate::SixelEncoder) struct to dither the result.
pub trait Dither: private::Sealed {
    const KERNEL: &[(isize, isize, f32)];
    const DIV: f32;

    /// Take the input image and convert it to the input palette, applying a
    /// dithering algorithm to the result.
    fn dither_and_palettize(image: &RgbImage, in_palette: &[Lab]) -> Vec<usize> {
        let pixels = image.pixels().copied().map(rgb_to_lab).collect::<Vec<_>>();

        let mut palette = KdTree::<_, _, 3, 257, u32>::with_capacity(in_palette.len());

        for (idx, color) in in_palette.iter().enumerate() {
            palette.add(color.as_ref(), idx);
        }

        let mut result = vec![0; image.width() as usize * image.height() as usize];

        let mut spills = vec![[0.0; 3]; image.width() as usize * image.height() as usize];

        for (idx, p) in pixels.iter().copied().enumerate() {
            let pixel = <Lab>::new(
                p.l + spills[idx][0],
                p.a + spills[idx][1],
                p.b + spills[idx][2],
            );

            let palette_idx =
                palette.nearest_one::<kiddo::SquaredEuclidean>(&[pixel.l, pixel.a, pixel.b]);

            let error0 = pixel.l - in_palette[palette_idx.item].l;
            let error1 = pixel.a - in_palette[palette_idx.item].a;
            let error2 = pixel.b - in_palette[palette_idx.item].b;
            let spill = [error0, error1, error2];

            result[idx] = palette_idx.item;

            for (dx, dy, m) in Self::KERNEL {
                let x = idx as isize % image.width() as isize + dx;
                let y = idx as isize / image.width() as isize + dy;
                if x < 0 || y < 0 || x >= image.width() as isize || y >= image.height() as isize {
                    continue;
                }

                let jdx = y * image.width() as isize + x;
                let target = &mut spills[jdx as usize];
                *target = [
                    target[0] + (spill[0] * m) / Self::DIV / 2.0,
                    target[1] + (spill[1] * m) / Self::DIV / 2.0,
                    target[2] + (spill[2] * m) / Self::DIV / 2.0,
                ];
            }
        }

        result
    }
}

/// Do not perform dithering. This is the fastest possible palette encoding
/// option.
pub struct NoDither;
impl private::Sealed for NoDither {}
impl Dither for NoDither {
    const DIV: f32 = 1.0;
    const KERNEL: &[(isize, isize, f32)] = &[];

    fn dither_and_palettize(image: &RgbImage, in_palette: &[Lab]) -> Vec<usize> {
        let pixels = image.pixels().copied().map(rgb_to_lab).collect::<Vec<_>>();

        let mut palette = KdTree::<_, _, 3, 32, u32>::with_capacity(in_palette.len());

        for (idx, color) in in_palette.iter().enumerate() {
            palette.add(color.as_ref(), idx);
        }

        let mut result = vec![0; image.width() as usize * image.height() as usize];
        pixels
            .into_par_iter()
            .zip(&mut result)
            .for_each(|(p, dest)| {
                *dest = palette
                    .nearest_one::<kiddo::SquaredEuclidean>(&[p.l, p.a, p.b])
                    .item;
            });

        result
    }
}

pub struct Sierra;
impl private::Sealed for Sierra {}
impl Dither for Sierra {
    const DIV: f32 = 32.0;
    // _ _ X 5 3
    // 2 4 5 4 2
    // _ 2 3 2 _
    const KERNEL: &[(isize, isize, f32)] = &[
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
}

pub struct Sierra2;
impl private::Sealed for Sierra2 {}
impl Dither for Sierra2 {
    const DIV: f32 = 16.0;
    // _ _ X 4 3
    // 1 2 3 2 1
    // _ 1 2 1 _
    const KERNEL: &[(isize, isize, f32)] = &[
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
}

pub struct SierraLite;
impl private::Sealed for SierraLite {}
impl Dither for SierraLite {
    const DIV: f32 = 4.0;
    //  _ X 2
    //  1 1 _
    const KERNEL: &[(isize, isize, f32)] = &[
        //x y  m
        (1, 0, 2.0),
        //
        (-1, 1, 1.0),
        (0, 1, 1.0),
    ];
}

pub struct FloydSteinberg;
impl private::Sealed for FloydSteinberg {}
impl Dither for FloydSteinberg {
    const DIV: f32 = 16.0;
    // _ X 7
    // 3 5 3
    const KERNEL: &[(isize, isize, f32)] = &[
        //x y  m
        (1, 0, 7.0),
        //
        (-1, 1, 3.0),
        (0, 1, 5.0),
        (1, 1, 1.0),
    ];
}

pub struct JJN;
impl private::Sealed for JJN {}
impl Dither for JJN {
    const DIV: f32 = 48.0;
    // _ _ X 7 5
    // 3 5 7 5 3
    // 1 2 5 3 1
    const KERNEL: &[(isize, isize, f32)] = &[
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
}

pub struct Stucki;
impl private::Sealed for Stucki {}
impl Dither for Stucki {
    const DIV: f32 = 42.0;
    // _ _ X 8 4
    // 2 4 8 4 2
    // 1 2 4 2 1
    const KERNEL: &[(isize, isize, f32)] = &[
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
}

pub struct Atkinson;
impl private::Sealed for Atkinson {}
impl Dither for Atkinson {
    const DIV: f32 = 8.0;
    // _ X 1 1
    // 1 1 1 _
    // _ 1 _ _
    const KERNEL: &[(isize, isize, f32)] = &[
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
}

pub struct Burkes;
impl private::Sealed for Burkes {}
impl Dither for Burkes {
    const DIV: f32 = 32.0;
    // _ _ X 8 4
    // 2 4 8 4 2
    const KERNEL: &[(isize, isize, f32)] = &[
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
}

pub struct Bayer;
impl private::Sealed for Bayer {}
impl Dither for Bayer {
    const DIV: f32 = 0.0;
    const KERNEL: &[(isize, isize, f32)] = &[];

    fn dither_and_palettize(image: &RgbImage, in_palette: &[Lab]) -> Vec<usize> {
        let matrix_size = image.width().max(image.height()).ilog2().max(2) as usize;
        let mut matrix = vec![0.0; matrix_size * matrix_size];

        (0..matrix_size as u32)
            .into_par_iter()
            .zip(matrix.par_chunks_mut(matrix_size))
            .for_each(|(y, row)| {
                for (x, cell) in row.iter_mut().enumerate() {
                    let x = x as u32;
                    let bits = (x ^ y).dilate_expand::<2>().value()
                        | (y.dilate_expand::<2>().value() << 1);
                    let bits = bits.reverse_bits() >> bits.leading_zeros();
                    *cell = bits as f32 / matrix_size.pow(2) as f32;
                }
            });

        let mut palette = KdTree::<_, _, 3, 257, u32>::with_capacity(in_palette.len());
        for (idx, color) in in_palette.iter().enumerate() {
            palette.add(color.as_ref(), idx);
        }

        let mut result = vec![0; image.width() as usize * image.height() as usize];
        let pixels = image.pixels().copied().map(rgb_to_lab).collect::<Vec<_>>();

        pixels
            .into_par_iter()
            .enumerate()
            .zip(&mut result)
            .for_each(|((idx, p), dest)| {
                let x = (idx % image.width() as usize) % matrix_size;
                let y = (idx / image.width() as usize) % matrix_size;

                let [l1, l2] = palette
                    .nearest_n::<kiddo::SquaredEuclidean>(p.as_ref(), 2)
                    .try_into()
                    .unwrap();

                let p_dist = in_palette[l1.item].distance_squared(in_palette[l2.item]);
                let t = ((l1.distance - l2.distance + p_dist) / (2.0 * p_dist)).clamp(0.0, 1.0);

                let m_idx = x + y * matrix_size;
                if t > matrix[m_idx] {
                    *dest = l2.item;
                } else {
                    *dest = l1.item;
                }
            });

        result
    }
}

/// Uses the Sobol sequence to perturb the colors in a quasi-random manner. This
/// is somewhere between ordered dithering and stochastic dithering. The results
/// are generally decent, although they may appear textured like paper.
pub struct Sobol;
impl private::Sealed for Sobol {}
impl Dither for Sobol {
    const DIV: f32 = 0.0;
    const KERNEL: &[(isize, isize, f32)] = &[];

    fn dither_and_palettize(image: &RgbImage, in_palette: &[Lab]) -> Vec<usize> {
        let r = (1usize << 8).div_ceil(in_palette.len() * in_palette.len().ilog2() as usize) as f32;

        let mut palette = KdTree::<_, _, 3, 257, u32>::with_capacity(in_palette.len());
        for (idx, color) in in_palette.iter().enumerate() {
            palette.add(color.as_ref(), idx);
        }

        let mut result = vec![0; image.width() as usize * image.height() as usize];
        let pixels = image.pixels().copied().map(rgb_to_lab).collect::<Vec<_>>();

        pixels
            .into_par_iter()
            .enumerate()
            .zip(&mut result)
            .for_each(|((idx, p), dest)| {
                let [l, a, b, _] = sample_4d(idx as u32 % (1 << 16), 0, idx as u32 / (1 << 16));

                let l = p.l + (l - 0.5) * 2.0 * r;
                let a = p.a + (a - 0.5) * 2.0 * r;
                let b = p.b + (b - 0.5) * 2.0 * r;

                *dest = palette
                    .nearest_one::<kiddo::SquaredEuclidean>(&[l, a, b])
                    .item;
            });

        result
    }
}
