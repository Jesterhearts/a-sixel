//! A-Sixel library for encoding sixel images.
//!
//! ### Basic Usage
//!
//! ```rust
//! use a_sixel::ADUSixelEncoder;
//! use image::RgbImage;
//!
//! let img = RgbImage::new(100, 100);
//! println!("{}", <ADUSixelEncoder>::encode(&img));
//! ```
//!
//! ## Choosing an Encoder
//! - I want fast encoding with good quality:
//!   - Use [`ADUSixelEncoder`]
//! - I'm time constrained:
//!   - Use [`ADUSixelEncoder`], [`OctreeSixelEncoder`], or [`BitSixelEncoder`].
//!     You can customize `ADU` by lowering the `STEPS` parameter to run faster
//!     if necessary while still getting good results.
//! - I'm _really_ time constrained and can sacrifice a little quality:
//!   - Use [`BitSixelEncoder<NoDither>`].
//! - I want high quality encoding, and don't mind a bit more computation:
//!   - Use [`FocalSixelEncoder`].
//!   - This matters a lot less if you're not crunching the palette down below
//!     256 colors.
//!   - Note that this an experimental encoder. It will *likely* produce better
//!     results than just [`ADUSixelEncoder`], but it may not always do so. On
//!     the test images, for my personal preferences, I think it's slightly
//!     better - particularly at small palette sizes.
//!
//!     <details>
//!     <summary>How it works</summary>
//!
//!     Under the hood, it is a modified version of the `ADUSixelEncoder`
//!     that uses a weighted selection algorithm for its sample pixels. These
//!     weights are determined based on saliency maps and measures of
//!     statistical noise in the image.
//!
//!     In addition to the weighted selection, the distance metric used to
//!     determine which cluster to place a pixel into also incorporates the
//!     weight. Similar pixels with different weights will be nudged towards
//!     clusters with similar weights. This is a mild effect, but it seems
//!     to improve things over basic clustering when there are a lot of similar
//!     colors in an image.
//!
//!     </details>

mod adu;
mod bit;
pub mod dither;
mod focal;
mod kmeans;
mod median_cut;
mod octree;

use std::fmt::Write;

use image::{
    Rgb,
    RgbImage,
};
use palette::{
    encoding::Srgb,
    Hsl,
    IntoColor,
    Lab,
};
use rayon::{
    iter::{
        IndexedParallelIterator,
        ParallelIterator,
    },
    slice::ParallelSlice,
};

use crate::dither::{
    Dither,
    Sierra,
};
pub use crate::{
    adu::ADUPaletteBuilder,
    bit::BitPaletteBuilder,
    focal::FocalPaletteBuilder,
    kmeans::KMeansPaletteBuilder,
    median_cut::MedianCutPaletteBuilder,
    octree::OctreePaletteBuilder,
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

/// A trait for types that perform quantization of an image to a target palette
/// size.
pub trait PaletteBuilder: private::Sealed {
    const PALETTE_SIZE: usize;

    /// Take in an image and return a quantized palette based on the colors in
    /// the image. The returned vector may be `<= PALETTE_SIZE` in length.
    fn build_palette(image: &RgbImage) -> Vec<Lab>;
}

const fn num2six(num: u8) -> char {
    (0x3f + num) as char
}

/// The main type for performing sixel encoding.
///
/// It is provided with two generic parameters:
/// - A [`PaletteBuilder`] to generate a color palette from the input image
///   (sixel only supports up to 256 colors).
/// - A [`Dither`] type to apply dithering to the reduced color image before
///   encoding it into sixel format.
///
/// A number of type aliases are provided for common configurations, such as
/// [`ADUSixelEncoder256`], which uses the [`ADUPaletteBuilder`] with 256
/// colors.
///
/// # Choosing a `PaletteBuilder`
/// - [`ADUPaletteBuilder`] is a good default choice for minimizing the error
///   across the image. For maximum color accuracy at the cost of speed, you can
///   use [`ADUSixelEncoder256High`].
/// - [`FocalPaletteBuilder`] is a good choice if the image has highlights and
///   other color details that ADU might squash, but is experimental and much
///   slower. You can increase the number of steps to improve accuracy even
///   futher, as is done by [`FocalSixelEncoder256High`].
/// - Other palette builders are available, but are likely to perform less well
///   at image accuracy than either of these two.
///
/// # Choosing a `Dither`
/// - [`Sierra`] is a good default choice for dithering, as it produces
///   high-quality results with minimal artifacts.
/// - [`NoDither`](dither::NoDither) can be used if performance is a concern.
pub struct SixelEncoder<P: PaletteBuilder = FocalPaletteBuilder, D: Dither = Sierra> {
    _p: std::marker::PhantomData<P>,
    _d: std::marker::PhantomData<D>,
}

pub type ADUSixelEncoder8<D = Sierra> = SixelEncoder<ADUPaletteBuilder<8>, D>;
pub type ADUSixelEncoder16<D = Sierra> = SixelEncoder<ADUPaletteBuilder<16>, D>;
pub type ADUSixelEncoder32<D = Sierra> = SixelEncoder<ADUPaletteBuilder<32>, D>;
pub type ADUSixelEncoder64<D = Sierra> = SixelEncoder<ADUPaletteBuilder<64>, D>;
pub type ADUSixelEncoder128<D = Sierra> = SixelEncoder<ADUPaletteBuilder<128>, D>;
pub type ADUSixelEncoder256<D = Sierra> = SixelEncoder<ADUPaletteBuilder<256>, D>;
pub type ADUSixelEncoder256High<D = Sierra> = SixelEncoder<ADUPaletteBuilder, D>;
pub type ADUSixelEncoder<D = Sierra> = ADUSixelEncoder256<D>;

pub type FocalSixelEncoderMono<D = Sierra> = SixelEncoder<FocalPaletteBuilder<2>, D>;
pub type FocalSixelEncoder4<D = Sierra> = SixelEncoder<FocalPaletteBuilder<4>, D>;
pub type FocalSixelEncoder8<D = Sierra> = SixelEncoder<FocalPaletteBuilder<8>, D>;
pub type FocalSixelEncoder16<D = Sierra> = SixelEncoder<FocalPaletteBuilder<16>, D>;
pub type FocalSixelEncoder32<D = Sierra> = SixelEncoder<FocalPaletteBuilder<32>, D>;
pub type FocalSixelEncoder64<D = Sierra> = SixelEncoder<FocalPaletteBuilder<64>, D>;
pub type FocalSixelEncoder128<D = Sierra> = SixelEncoder<FocalPaletteBuilder<128>, D>;
pub type FocalSixelEncoder256<D = Sierra> = SixelEncoder<FocalPaletteBuilder<256>, D>;
pub type FocalSixelEncoder256High<D = Sierra> =
    SixelEncoder<FocalPaletteBuilder<256, { 1 << 12 }, { 1 << 22 }>, D>;
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

pub type BitSixelEncoderMono<D = Sierra> = SixelEncoder<BitPaletteBuilder<2>, D>;
pub type BitSixelEncoder4<D = Sierra> = SixelEncoder<BitPaletteBuilder<4>, D>;
pub type BitSixelEncoder8<D = Sierra> = SixelEncoder<BitPaletteBuilder<8>, D>;
pub type BitSixelEncoder16<D = Sierra> = SixelEncoder<BitPaletteBuilder<16>, D>;
pub type BitSixelEncoder32<D = Sierra> = SixelEncoder<BitPaletteBuilder<32>, D>;
pub type BitSixelEncoder64<D = Sierra> = SixelEncoder<BitPaletteBuilder<64>, D>;
pub type BitSixelEncoder128<D = Sierra> = SixelEncoder<BitPaletteBuilder<128>, D>;
pub type BitSixelEncoder256<D = Sierra> = SixelEncoder<BitPaletteBuilder<256>, D>;
pub type BitSixelEncoder<D = Sierra> = BitSixelEncoder256<D>;

pub type OctreeSixelEncoderMono<D = Sierra, const USE_MIN_HEAP: bool = false> =
    SixelEncoder<OctreePaletteBuilder<2, USE_MIN_HEAP>, D>;
pub type OctreeSixelEncoder4<D = Sierra, const USE_MIN_HEAP: bool = false> =
    SixelEncoder<OctreePaletteBuilder<4, USE_MIN_HEAP>, D>;
pub type OctreeSixelEncoder8<D = Sierra, const USE_MIN_HEAP: bool = false> =
    SixelEncoder<OctreePaletteBuilder<8, USE_MIN_HEAP>, D>;
pub type OctreeSixelEncoder16<D = Sierra, const USE_MIN_HEAP: bool = false> =
    SixelEncoder<OctreePaletteBuilder<16, USE_MIN_HEAP>, D>;
pub type OctreeSixelEncoder32<D = Sierra, const USE_MIN_HEAP: bool = false> =
    SixelEncoder<OctreePaletteBuilder<32, USE_MIN_HEAP>, D>;
pub type OctreeSixelEncoder64<D = Sierra, const USE_MIN_HEAP: bool = false> =
    SixelEncoder<OctreePaletteBuilder<64, USE_MIN_HEAP>, D>;
pub type OctreeSixelEncoder128<D = Sierra, const USE_MIN_HEAP: bool = false> =
    SixelEncoder<OctreePaletteBuilder<128, USE_MIN_HEAP>, D>;
pub type OctreeSixelEncoder256<D = Sierra, const USE_MIN_HEAP: bool = false> =
    SixelEncoder<OctreePaletteBuilder<256, USE_MIN_HEAP>, D>;
pub type OctreeSixelEncoder<D = Sierra, const USE_MIN_HEAP: bool = false> =
    OctreeSixelEncoder256<D, USE_MIN_HEAP>;

pub type KMeansSixelEncoderMono<D = Sierra> = SixelEncoder<KMeansPaletteBuilder<2>, D>;
pub type KMeansSixelEncoder4<D = Sierra> = SixelEncoder<KMeansPaletteBuilder<4>, D>;
pub type KMeansSixelEncoder8<D = Sierra> = SixelEncoder<KMeansPaletteBuilder<8>, D>;
pub type KMeansSixelEncoder16<D = Sierra> = SixelEncoder<KMeansPaletteBuilder<16>, D>;
pub type KMeansSixelEncoder32<D = Sierra> = SixelEncoder<KMeansPaletteBuilder<32>, D>;
pub type KMeansSixelEncoder64<D = Sierra> = SixelEncoder<KMeansPaletteBuilder<64>, D>;
pub type KMeansSixelEncoder128<D = Sierra> = SixelEncoder<KMeansPaletteBuilder<128>, D>;
pub type KMeansSixelEncoder256<D = Sierra> = SixelEncoder<KMeansPaletteBuilder<256>, D>;
pub type KMeansSixelEncoder<D = Sierra> = KMeansSixelEncoder256<D>;

impl<P: PaletteBuilder, D: Dither> SixelEncoder<P, D> {
    pub fn encode(image: RgbImage) -> String {
        let palette = P::build_palette(&image);

        let mut sixel_string = r#"Pq"1;1;"#.to_string();
        sixel_string
            .write_fmt(format_args!("{};{}", image.height(), image.width()))
            .expect("Failed to write sixel bounds");

        for (i, lab) in palette.iter().copied().enumerate() {
            let hsl: Hsl = lab.into_color();
            // Sixel hue is offset by 120 degrees from the common hue values.
            let deg = (hsl.hue.into_positive_degrees().round() as u16 + 120) % 360;

            sixel_string
                .write_fmt(format_args!(
                    "#{i};1;{deg};{};{}",
                    (hsl.lightness * 100.0).round() as u8,
                    (hsl.saturation * 100.0).round() as u8,
                ))
                .expect("Failed to palette entry");
        }

        let paletted_pixels = D::dither_and_palettize(&image, &palette, P::PALETTE_SIZE);

        let rows: Vec<&[usize]> = paletted_pixels
            .chunks(image.width() as usize)
            .collect::<Vec<_>>();

        let mut strings = vec![String::new(); rows.len().div_ceil(6)];
        rows.par_chunks(6)
            .zip(&mut strings)
            .for_each(|(stack, sixel_string)| {
                let mut row_palette = vec![false; P::PALETTE_SIZE];
                row_palette.fill(false);
                for idx in
                    stack_iter(stack).flat_map(|(((((zero, one), two), three), four), five)| {
                        std::iter::once(zero)
                            .chain(one)
                            .chain(two)
                            .chain(three)
                            .chain(four)
                            .chain(five)
                    })
                {
                    row_palette[idx] = true;
                }

                for (color, _) in row_palette.iter().copied().enumerate().filter(|(_, v)| *v) {
                    let mut stack_string = SixelRow::new(sixel_string, color);
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
            });

        sixel_string.extend(strings);
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

fn stack_iter<'a>(stack: &'a [&[usize]]) -> impl Iterator<Item = StackTuple> + 'a {
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
