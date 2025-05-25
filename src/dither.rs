use core::f32;

use image::RgbImage;
use kiddo::float::kdtree::KdTree;
use palette::Lab;

use crate::{
    private,
    rgb_to_lab,
};

/// https://tannerhelland.com/2012/12/28/dithering-eleven-algorithms-source-code.html
pub trait Dither: private::Sealed {
    const KERNEL: &[(isize, isize, f32)];
    const DIV: f32;

    fn dither_and_palettize(image: &RgbImage, in_palette: &[Lab]) -> Vec<usize> {
        let pixels = image.pixels().copied().map(rgb_to_lab).collect::<Vec<_>>();

        let mut palette = KdTree::<_, _, 3, 32, u32>::with_capacity(in_palette.len());

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

pub struct NoDither;
impl private::Sealed for NoDither {}
impl Dither for NoDither {
    const DIV: f32 = 1.0;
    const KERNEL: &[(isize, isize, f32)] = &[];
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
