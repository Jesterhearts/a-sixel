mod adu;
pub mod dither;
mod focal;
mod median_cut;

use std::fmt::Write;

use image::{
    Rgb,
    RgbImage,
};
use palette::{
    Hsl,
    IntoColor,
    Lab,
    encoding::Srgb,
};

use crate::dither::{
    Dither,
    Sierra,
};
pub use crate::{
    adu::ADUPaletteBuilder,
    focal::FocalPaletteBuilder,
    median_cut::MedianCutPaletteBuilder,
};

struct SixelRow<'c> {
    committed: &'c mut String,
    pending: char,
    count: usize,
}

impl<'c> SixelRow<'c> {
    fn new(builder: &'c mut String, color: usize) -> Self {
        builder
            .write_fmt(format_args!("#{color}"))
            .expect("Failed to write color selector");

        Self {
            committed: builder,
            pending: num2six(0),
            count: 0,
        }
    }

    fn push(&mut self, ch: char) {
        if ch == self.pending {
            self.count += 1;
        } else {
            self.commit();
            self.pending = ch;
            self.count = 1;
        }
    }

    fn commit(&mut self) {
        if self.count > 3 {
            self.committed
                .write_fmt(format_args!("!{}{}", self.count, self.pending))
                .expect("Failed to write to string");
        } else {
            for _ in 0..self.count {
                self.committed.push(self.pending);
            }
        }
    }

    fn finalize(mut self) {
        self.commit();
        self.committed.push('$');
    }
}

mod private {
    pub trait Sealed {}
}

pub trait PaletteBuilder: private::Sealed {
    const PALETTE_SIZE: usize;

    fn build_palette(image: &RgbImage) -> Vec<Lab>;
}

const fn num2six(num: u8) -> char {
    (0x3f + num) as char
}

pub struct SixelEncoder<P: PaletteBuilder = FocalPaletteBuilder, D: Dither = Sierra> {
    _p: std::marker::PhantomData<P>,
    _d: std::marker::PhantomData<D>,
}

pub type ADUSixelEncoder8<D = Sierra> = SixelEncoder<ADUPaletteBuilder<8, 1, { 1 << 17 }>, D>;
pub type ADUSixelEncoder16<D = Sierra> = SixelEncoder<ADUPaletteBuilder<16, 1, { 1 << 17 }>, D>;
pub type ADUSixelEncoder32<D = Sierra> = SixelEncoder<ADUPaletteBuilder<32, 2, { 1 << 17 }>, D>;
pub type ADUSixelEncoder64<D = Sierra> = SixelEncoder<ADUPaletteBuilder<64, 4, { 1 << 17 }>, D>;
pub type ADUSixelEncoder128<D = Sierra> = SixelEncoder<ADUPaletteBuilder<128, 8, { 1 << 17 }>, D>;
pub type ADUSixelEncoder256<D = Sierra> = SixelEncoder<ADUPaletteBuilder<256, 16, { 1 << 17 }>, D>;
pub type ADUSixelEncoder<D = Sierra> = ADUSixelEncoder256<D>;

pub type FocalSixelEncoderMono<D = Sierra> = SixelEncoder<FocalPaletteBuilder<2>, D>;
pub type FocalSixelEncoder4<D = Sierra> = SixelEncoder<FocalPaletteBuilder<4>, D>;
pub type FocalSixelEncoder8<D = Sierra> = SixelEncoder<FocalPaletteBuilder<8>, D>;
pub type FocalSixelEncoder16<D = Sierra> = SixelEncoder<FocalPaletteBuilder<16>, D>;
pub type FocalSixelEncoder32<D = Sierra> = SixelEncoder<FocalPaletteBuilder<32>, D>;
pub type FocalSixelEncoder64<D = Sierra> = SixelEncoder<FocalPaletteBuilder<64>, D>;
pub type FocalSixelEncoder128<D = Sierra> = SixelEncoder<FocalPaletteBuilder<128>, D>;
pub type FocalSixelEncoder256<D = Sierra> = SixelEncoder<FocalPaletteBuilder<256>, D>;
pub type FocalSixelEncoder<D = Sierra> = FocalSixelEncoder256<D>;

pub type MedianCutSixelEncoderMono<D = Sierra> = SixelEncoder<MedianCutPaletteBuilder<2>, D>;
pub type MedianCutSixelEncoder4<D = Sierra> = SixelEncoder<MedianCutPaletteBuilder<4>, D>;
pub type MedianCutSixelEncoder8<D = Sierra> = SixelEncoder<MedianCutPaletteBuilder<8>, D>;
pub type MedianCutSixelEncoder16<D = Sierra> = SixelEncoder<MedianCutPaletteBuilder<16>, D>;
pub type MedianCutSixelEncoder32<D = Sierra> = SixelEncoder<MedianCutPaletteBuilder<32>, D>;
pub type MedianCutSixelEncoder64<D = Sierra> = SixelEncoder<MedianCutPaletteBuilder<64>, D>;
pub type MedianCutSixelEncoder128<D = Sierra> = SixelEncoder<MedianCutPaletteBuilder<128>, D>;
pub type MedianCutSixelEncoder256<D = Sierra> = SixelEncoder<MedianCutPaletteBuilder<256>, D>;
pub type MedianCutSixelEncoder<D = Sierra> = MedianCutSixelEncoder256<D>;

impl<P: PaletteBuilder, D: Dither> SixelEncoder<P, D> {
    pub fn encode(image: RgbImage) -> String {
        let palette = P::build_palette(&image);
        let mut sixel_string = r#"Pq"1;1;"#.to_string();
        sixel_string
            .write_fmt(format_args!("{};{}", image.height(), image.width()))
            .expect("Failed to write sixel bounds");

        for (i, lab) in palette.iter().copied().enumerate() {
            let hsl: Hsl = lab.into_color();
            // This may be a windows specific bug, but hue is offset by 120 degrees.
            let deg = (hsl.hue.into_positive_degrees().round() as u16 + 120) % 360;

            sixel_string
                .write_fmt(format_args!(
                    "#{i};1;{deg};{};{}",
                    (hsl.lightness * 100.0).round() as u8,
                    (hsl.saturation * 100.0).round() as u8,
                ))
                .expect("Failed to palette entry");
        }

        let paletted_pixels = D::dither_and_palettize(&image, &palette);

        let rows: Vec<&[usize]> = paletted_pixels
            .chunks(image.width() as usize)
            .collect::<Vec<_>>();

        let mut row_palette = vec![false; P::PALETTE_SIZE];
        for stack in rows.chunks(6) {
            row_palette.fill(false);
            for idx in stack_iter(stack).flat_map(|(((((zero, one), two), three), four), five)| {
                std::iter::once(zero)
                    .chain(one)
                    .chain(two)
                    .chain(three)
                    .chain(four)
                    .chain(five)
            }) {
                row_palette[idx] = true;
            }

            for (color, _) in row_palette.iter().copied().enumerate().filter(|(_, v)| *v) {
                let mut stack_string = SixelRow::new(&mut sixel_string, color);
                for (((((zero, one), two), three), four), five) in stack_iter(stack) {
                    let bits = (zero == color) as u8
                        | ((one == Some(color)) as u8) << 1
                        | ((two == Some(color)) as u8) << 2
                        | ((three == Some(color)) as u8) << 3
                        | ((four == Some(color)) as u8) << 4
                        | ((five == Some(color)) as u8) << 5;
                    let char = num2six(bits);
                    stack_string.push(char);
                }
                stack_string.finalize();
            }
            sixel_string.push('-');
        }

        sixel_string.push_str(r#"\"#);
        sixel_string
    }
}

type StackTuple = (
    (
        (((usize, Option<usize>), Option<usize>), Option<usize>),
        Option<usize>,
    ),
    Option<usize>,
);

fn stack_iter(stack: &[&[usize]]) -> impl Iterator<Item = StackTuple> {
    stack
        .first()
        .into_iter()
        .cloned()
        .flatten()
        .copied()
        .zip(
            stack
                .get(1)
                .into_iter()
                .cloned()
                .flatten()
                .copied()
                .map(Some)
                .chain(std::iter::repeat(None)),
        )
        .zip(
            stack
                .get(2)
                .into_iter()
                .cloned()
                .flatten()
                .copied()
                .map(Some)
                .chain(std::iter::repeat(None)),
        )
        .zip(
            stack
                .get(3)
                .into_iter()
                .cloned()
                .flatten()
                .copied()
                .map(Some)
                .chain(std::iter::repeat(None)),
        )
        .zip(
            stack
                .get(4)
                .into_iter()
                .cloned()
                .flatten()
                .copied()
                .map(Some)
                .chain(std::iter::repeat(None)),
        )
        .zip(
            stack
                .get(5)
                .into_iter()
                .cloned()
                .flatten()
                .copied()
                .map(Some)
                .chain(std::iter::repeat(None)),
        )
}

fn rgb_to_lab(Rgb([r, g, b]): Rgb<u8>) -> Lab {
    palette::rgb::Rgb::<Srgb, _>::new(r, g, b)
        .into_format::<f32>()
        .into_color()
}
