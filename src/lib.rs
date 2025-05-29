//! A-Sixel library for encoding sixel images.
//!
//! ### Basic Usage
//!
//! ```rust
//! use a_sixel::KMeansSixelEncoder;
//! use image::RgbImage;
//!
//! let img = RgbImage::new(100, 100);
//! println!("{}", <KMeansSixelEncoder>::encode(&img));
//! ```
//!
//! ## Choosing an Encoder
//! - I want fast encoding with good quality:
//!   - Use [`KMeansSixelEncoder`] or [`ADUSixelEncoder`].
//! - I'm time constrained:
//!   - Use [`ADUSixelEncoder`], [`BitSixelEncoder`], or [`OctreeSixelEncoder`].
//!     You can customize `ADU` by lowering the `STEPS` parameter to run faster
//!     if necessary while still getting good results.
//! - I'm _really_ time constrained and can sacrifice a little quality:
//!   - Use [`BitSixelEncoder<NoDither>`].
//! - I want high quality encoding, and don't mind a bit more computation:
//!   - Use [`KMediansSixelEncoder`] or [`FocalSixelEncoder`]. These can be
//!     _very_ slow on large images, but produce the highest quality results.
//!   - This matters a lot less if you're not crunching the palette down below
//!     256 colors. At 256 colors, most encoders will produce decent results.
//!
//! For a more detailed breakdown, here's the top encoders by average speed and
//! quality against the test images (speed figures will vary by machine):
//!
//! |    Algorithm         |    MSE        |    DSSIM     |    Execution Time (ms)    |
//! |:---------------------|:-------------:|:------------:|--------------------------:|
//! |    bit               |    34.06      |    0.0124    |     332                   |
//! |    bit-merge         |    16.30      |    0.0059    |     492                   |
//! |    bit-merge-better  |    14.07      |    0.0053    |     732                   |
//! |    bit-merge-best    |    12.17      |    0.0043    |    2120                   |
//! |    adu               |    15.56      |    0.0054    |    1407                   |
//! |    k-means           |    10.86      |    0.0040    |    6273                   |

pub mod adu;
pub mod bit;
pub mod bitmerge;
pub mod dither;
pub mod focal;
pub mod kmeans;
pub mod kmedians;
pub mod median_cut;
pub mod octree;

use std::{
    fmt::Write,
    sync::atomic::{
        AtomicBool,
        Ordering,
    },
};

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
        IntoParallelRefIterator,
        ParallelIterator,
    },
    slice::ParallelSlice,
};

pub use crate::{
    adu::ADUPaletteBuilder,
    bit::BitPaletteBuilder,
    bitmerge::BitMergePaletteBuilder,
    focal::FocalPaletteBuilder,
    kmeans::KMeansPaletteBuilder,
    kmedians::KMediansPaletteBuilder,
    median_cut::MedianCutPaletteBuilder,
    octree::OctreePaletteBuilder,
};
use crate::{
    adu::ADUSixelEncoder256,
    bit::BitSixelEncoder256,
    bitmerge::BitMergeSixelEncoder256,
    dither::{
        Dither,
        Sierra,
    },
    focal::FocalSixelEncoder256,
    kmeans::KMeansSixelEncoder256,
    kmedians::KMediansSixelEncoder256,
    median_cut::MedianCutSixelEncoder256,
    octree::OctreeSixelEncoder256,
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
    const NAME: &'static str;

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
/// - [`ADUPaletteBuilder`] or [`KMeansPaletteBuilder`] are good default choices
///   for minimizing the error across the image.
/// - [`KMediansPaletteBuilder`] is a higher-quality alternative to
///   [`KMeansPaletteBuilder`], but is slower.
/// - [`FocalPaletteBuilder`] is a good choice if the image has highlights and
///   other color details that ADU might squash, but is experimental. It is a
///   weighted k-medians implementation.
/// - Other palette builders are available, but are likely to perform less well
///   at image accuracy than these choices.
///
/// # Choosing a `Dither`
/// - [`Sierra`] is a good default choice for dithering, as it produces
///   high-quality results with minimal artifacts.
/// - [`NoDither`](dither::NoDither) can be used if performance is a concern.
pub struct SixelEncoder<P: PaletteBuilder = FocalPaletteBuilder<256>, D: Dither = Sierra> {
    _p: std::marker::PhantomData<P>,
    _d: std::marker::PhantomData<D>,
}

pub type ADUSixelEncoder<D = Sierra> = ADUSixelEncoder256<D>;
pub type BitMergeSixelEncoder<D = Sierra> = BitMergeSixelEncoder256<D>;
pub type BitMergeSixelEncoderBetter<D = Sierra> = BitMergeSixelEncoder256<D, 2048>;
pub type BitMergeSixelEncoderBest<D = Sierra> = BitMergeSixelEncoder256<D, 4096>;
pub type BitSixelEncoder<D = Sierra> = BitSixelEncoder256<D>;
pub type FocalSixelEncoder<D = Sierra> = FocalSixelEncoder256<D>;
pub type KMeansSixelEncoder<D = Sierra> = KMeansSixelEncoder256<D>;
pub type KMediansSixelEncoder<D = Sierra> = KMediansSixelEncoder256<D>;
pub type MedianCutSixelEncoder<D = Sierra> = MedianCutSixelEncoder256<D>;
pub type OctreeSixelEncoder<D = Sierra, const USE_MIN_HEAP: bool = false> =
    OctreeSixelEncoder256<D, USE_MIN_HEAP>;

impl<P: PaletteBuilder, D: Dither> SixelEncoder<P, D> {
    pub fn encode(image: &RgbImage) -> String {
        let palette = P::build_palette(image);

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

        let paletted_pixels = D::dither_and_palettize(image, &palette, P::PALETTE_SIZE);

        #[cfg(feature = "dump_mse")]
        {
            let dequant = paletted_pixels
                .iter()
                .map(|&idx| palette[idx])
                .collect::<Vec<_>>();
            let mse = dequant
                .par_iter()
                .zip(image.par_pixels())
                .map(|(l, rgb)| {
                    use palette::color_difference::EuclideanDistance;
                    let lab = rgb_to_lab(*rgb);
                    lab.distance_squared(*l)
                })
                .sum::<f32>()
                / (image.width() * image.height()) as f32;

            println!("MSE: {:.2} ({} colors)", mse, P::PALETTE_SIZE);
        }

        #[cfg(feature = "dump_dssim")]
        {
            use dssim_core::Dssim;

            let dssim = Dssim::new();
            let image_pixels = image
                .pixels()
                .copied()
                .map(|Rgb([r, g, b])| rgb::RGB::new(r, g, b))
                .collect::<Vec<_>>();
            let orig = dssim
                .create_image_rgb(
                    &image_pixels,
                    image.width() as usize,
                    image.height() as usize,
                )
                .unwrap();

            let palette_pixels = paletted_pixels
                .iter()
                .map(|&idx| {
                    let lab = palette[idx];
                    let rgb: palette::Srgb = lab.into_color();
                    let rgb = rgb.into_format::<u8>();
                    rgb::RGB::new(rgb.red, rgb.green, rgb.blue)
                })
                .collect::<Vec<_>>();
            let new = dssim
                .create_image_rgb(
                    &palette_pixels,
                    image.width() as usize,
                    image.height() as usize,
                )
                .unwrap();

            let (dssim, _) = dssim.compare(&orig, &new);

            println!("DSSIM: {:.4} ({} colors)", dssim, P::PALETTE_SIZE);
        }

        #[cfg(feature = "dump_image")]
        {
            use std::hash::{
                BuildHasher,
                Hasher,
                RandomState,
            };

            let mut output_image = image::ImageBuffer::new(image.width(), image.height());
            for (pixel, &idx) in output_image.pixels_mut().zip(&paletted_pixels) {
                let lab = palette[idx];
                let rgb: palette::Srgb = lab.into_color();
                let rgb = rgb.into_format::<u8>();
                *pixel = Rgb([rgb.red, rgb.green, rgb.blue]);
            }
            let rand = BuildHasher::build_hasher(&RandomState::new()).finish();

            output_image
                .save(format!("{}-{rand}.png", P::NAME))
                .expect("Failed to save output image");
        }

        let rows: Vec<&[usize]> = paletted_pixels
            .chunks(image.width() as usize)
            .collect::<Vec<_>>();

        let mut strings = vec![String::new(); rows.len().div_ceil(6)];
        rows.par_chunks(6)
            .zip(&mut strings)
            .for_each(|(stack, sixel_string)| {
                let row_palette =
                    Vec::from_iter((0..P::PALETTE_SIZE).map(|_| AtomicBool::new(false)));
                stack
                    .par_iter()
                    .flat_map(|row| row.par_iter().copied())
                    .for_each(|idx| {
                        row_palette[idx].store(true, Ordering::Relaxed);
                    });

                for (color, _) in row_palette
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.load(Ordering::Relaxed))
                {
                    let mut stack_string = SixelRow::new(sixel_string, color);
                    for idx in 0..stack[0].len() {
                        let bits = (stack[0][idx] == color) as u8
                            | ((stack.get(1).map(|r| r[idx]) == Some(color)) as u8) << 1
                            | ((stack.get(2).map(|r| r[idx]) == Some(color)) as u8) << 2
                            | ((stack.get(3).map(|r| r[idx]) == Some(color)) as u8) << 3
                            | ((stack.get(4).map(|r| r[idx]) == Some(color)) as u8) << 4
                            | ((stack.get(5).map(|r| r[idx]) == Some(color)) as u8) << 5;
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

fn rgb_to_lab(Rgb([r, g, b]): Rgb<u8>) -> Lab {
    palette::rgb::Rgb::<Srgb, _>::new(r, g, b)
        .into_format::<f32>()
        .into_color()
}
