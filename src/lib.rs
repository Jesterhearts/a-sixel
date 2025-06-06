//! A-Sixel library for encoding sixel images.
//!
//! ### Basic Usage
//!
//! ```rust
//! use a_sixel::BitMergeSixelEncoderBest;
//! use image::RgbImage;
//!
//! let img = RgbImage::new(100, 100);
//! println!("{}", <BitMergeSixelEncoderBest>::encode(&img));
//! ```
//!
//! ## Choosing an Encoder
//! - I want good quality:
//!   - Use [`BitMergeSixelEncoderBest`] or [`KMeansSixelEncoder`].
//! - I'm time constrained:
//!   - Use [`BitMergeSixelEncoderLow`], [`BitSixelEncoder`], or
//!     [`OctreeSixelEncoder`].
//! - I'm _really_ time constrained and can sacrifice a little quality:
//!   - Use [`BitSixelEncoder<NoDither>`].
//!
//! For a more detailed breakdown, here's the encoders by average speed and
//! quality against the test images (speed figures will vary) at 256 colors with
//! Sierra dithering:
//!
//! | Algorithm        |   MSE | DSSIM  | PHash Distance | Mean ΔE | Max ΔE | ΔE >2.3 | ΔE >5.0 | Execution Time (ms) |
//! | :--------------- | ----: | :----: | -------------: | ------: | -----: | ------: | ------: | ------------------: |
//! | adu              | 15.56 | 0.0054 |           8.66 |    1.79 |  12.88 |     32% |    4.4% |                1473 |
//! | bit              | 36.42 | 0.0132 |          31.14 |    3.16 |  11.03 |     65% |   15.1% |                 367 |
//! | bit-merge-low    | 12.10 | 0.0046 |          13.97 |    1.95 |   9.98 |     32% |    2.2% |                 559 |
//! | bit-merge        | 10.96 | 0.0040 |          13.38 |    1.89 |  10.01 |     31% |    2.2% |                1198 |
//! | bit-merge-better | 10.77 | 0.0039 |          13.34 |    1.85 |  10.21 |     31% |    2.2% |                2196 |
//! | bit-merge-best   | 10.75 | 0.0039 |          13.58 |    1.85 |  10.21 |     31% |    2.2% |                2850 |
//! | focal            | 14.98 | 0.0057 |          20.21 |    2.30 |   9.16 |     45% |    3.3% |                2396 |
//! | k-means          | 10.86 | 0.0040 |          13.41 |    1.80 |  10.17 |     29% |    2.2% |                6208 |
//! | k-medians        | 18.68 | 0.0075 |          18.52 |    2.60 |  10.17 |     53% |    5.1% |               10688 |
//! | median-cut       | 20.27 | 0.0061 |          16.45 |    2.24 |  10.36 |     42% |    5.9% |                 627 |
//! | octree           | 66.60 | 0.0163 |          26.03 |    3.89 |  12.49 |     79% |   25.4% |                 589 |
//!
//! Here's the encoders at 16 colors with Sierra dithering:
//!
//! | Algorithm  |    MSE | DSSIM  | PHash Distance | Mean ΔE | Max ΔE | ΔE >2.3 | ΔE >5.0 | Execution Time (ms) |
//! | :--------- | -----: | :----: | -------------: | ------: | -----: | ------: | ------: | ------------------: |
//! | adu        | 120.45 | 0.0371 |          39.83 |    4.02 |  18.39 |     66% |     33% |                 280 |
//! | bit        | 182.07 | 0.0492 |          59.79 |    5.53 |  16.61 |     89% |     51% |                 247 |
//! | bit-merge  |  98.10 | 0.0307 |          41.79 |    3.96 |  16.41 |     68% |     31% |                 993 |
//! | focal      | 121.73 | 0.0362 |          48.97 |    4.36 |  16.90 |     72% |     34% |                2171 |
//! | k-means    |  97.02 | 0.0297 |          42.83 |    3.99 |  16.40 |     67% |     31% |                 664 |
//! | k-medians  | 171.20 | 0.0486 |          57.38 |    5.41 |  16.62 |     91% |     49% |                5792 |
//! | median-cut | 168.88 | 0.0381 |          45.28 |    4.68 |  16.72 |     74% |     42% |                 317 |
//! | octree     | 546.05 | 0.0922 |          75.07 |    7.69 |  18.87 |     98% |     74% |                 341 |

#[cfg(feature = "adu")]
pub mod adu;
pub mod bit;
#[cfg(feature = "bit-merge")]
pub mod bitmerge;
pub mod dither;
#[cfg(feature = "focal")]
pub mod focal;
#[cfg(feature = "k-means")]
pub mod kmeans;
#[cfg(feature = "k-medians")]
pub mod kmedians;
#[cfg(feature = "median-cut")]
pub mod median_cut;
#[cfg(feature = "octree")]
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

#[cfg(feature = "adu")]
pub use crate::adu::ADUPaletteBuilder;
#[cfg(feature = "adu")]
use crate::adu::ADUSixelEncoder256;
pub use crate::bit::BitPaletteBuilder;
#[cfg(feature = "bit-merge")]
pub use crate::bitmerge::BitMergePaletteBuilder;
#[cfg(feature = "bit-merge")]
use crate::bitmerge::BitMergeSixelEncoder256;
#[cfg(feature = "focal")]
pub use crate::focal::FocalPaletteBuilder;
#[cfg(feature = "focal")]
use crate::focal::FocalSixelEncoder256;
#[cfg(feature = "k-means")]
pub use crate::kmeans::KMeansPaletteBuilder;
#[cfg(feature = "k-means")]
use crate::kmeans::KMeansSixelEncoder256;
#[cfg(feature = "k-medians")]
pub use crate::kmedians::KMediansPaletteBuilder;
#[cfg(feature = "k-medians")]
use crate::kmedians::KMediansSixelEncoder256;
#[cfg(feature = "median-cut")]
pub use crate::median_cut::MedianCutPaletteBuilder;
#[cfg(feature = "median-cut")]
use crate::median_cut::MedianCutSixelEncoder256;
#[cfg(feature = "octree")]
pub use crate::octree::OctreePaletteBuilder;
#[cfg(feature = "octree")]
use crate::octree::OctreeSixelEncoder256;
use crate::{
    bit::BitSixelEncoder256,
    dither::{
        Dither,
        Sierra,
    },
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
/// - [`BitMergePaletteBuilder`] or [`KMeansPaletteBuilder`] are good default
///   choices for minimizing the error across the image.
/// - [`FocalPaletteBuilder`] is a good choice if the image has highlights and
///   other details that other encoders might squash, but is experimental. It is
///   a weighted k-means implementation. Depending on the image, `KMeans` may be
///   able to capture these highlights already, but it's worth trying if you're
///   trying to preserve specific image characteristics.
///
/// # Choosing a `Dither`
/// - [`Sierra`] is a good default choice for dithering, as it produces
///   high-quality results with minimal artifacts.
/// - [`NoDither`](dither::NoDither) can be used if performance is a concern.
pub struct SixelEncoder<P: PaletteBuilder = BitPaletteBuilder<256>, D: Dither = Sierra> {
    _p: std::marker::PhantomData<P>,
    _d: std::marker::PhantomData<D>,
}

#[cfg(feature = "adu")]
pub type ADUSixelEncoder<D = Sierra> = ADUSixelEncoder256<D>;
#[cfg(feature = "bit-merge")]
pub type BitMergeSixelEncoderLow<D = Sierra> = BitMergeSixelEncoder256<D, { 1 << 14 }>;
#[cfg(feature = "bit-merge")]
pub type BitMergeSixelEncoder<D = Sierra> = BitMergeSixelEncoder256<D>;
#[cfg(feature = "bit-merge")]
pub type BitMergeSixelEncoderBetter<D = Sierra> = BitMergeSixelEncoder256<D, { 1 << 20 }>;
#[cfg(feature = "bit-merge")]
pub type BitMergeSixelEncoderBest<D = Sierra> = BitMergeSixelEncoder256<D, { 1 << 21 }>;
pub type BitSixelEncoder<D = Sierra> = BitSixelEncoder256<D>;
#[cfg(feature = "focal")]
pub type FocalSixelEncoder<D = Sierra> = FocalSixelEncoder256<D>;
#[cfg(feature = "k-means")]
pub type KMeansSixelEncoder<D = Sierra> = KMeansSixelEncoder256<D>;
#[cfg(feature = "k-medians")]
pub type KMediansSixelEncoder<D = Sierra> = KMediansSixelEncoder256<D>;
#[cfg(feature = "median-cut")]
pub type MedianCutSixelEncoder<D = Sierra> = MedianCutSixelEncoder256<D>;
#[cfg(feature = "octree")]
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

        #[cfg(feature = "dump-mse")]
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

        #[cfg(feature = "dump-delta-e")]
        {
            let dequant = paletted_pixels
                .iter()
                .map(|&idx| palette[idx])
                .collect::<Vec<_>>();
            let differences = image
                .par_pixels()
                .copied()
                .zip(dequant.par_iter())
                .map(|(rgb, lab)| {
                    use palette::color_difference::ImprovedCiede2000;
                    let lab_rgb = rgb_to_lab(rgb);
                    lab_rgb.improved_difference(*lab)
                })
                .collect::<Vec<_>>();

            let mean_diff =
                differences.iter().sum::<f32>() / (image.width() * image.height()) as f32;
            let max_diff = differences.iter().copied().fold(0.0, f32::max);
            let two_three_threshold = differences.iter().copied().filter(|d| *d > 2.3).count()
                as f32
                / (image.width() * image.height()) as f32;
            let five_threshold = differences.iter().copied().filter(|d| *d > 5.0).count() as f32
                / (image.width() * image.height()) as f32;

            println!("Mean DeltaE: {:.2} ({} colors)", mean_diff, P::PALETTE_SIZE);
            println!("Max DeltaE: {:.2} ({} colors)", max_diff, P::PALETTE_SIZE);
            println!(
                "DeltaE > 2.3: {:.2} ({} colors)",
                two_three_threshold,
                P::PALETTE_SIZE
            );
            println!(
                "DeltaE > 5.0: {:.2} ({} colors)",
                five_threshold,
                P::PALETTE_SIZE
            );
        }

        #[cfg(feature = "dump-dssim")]
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

        #[cfg(feature = "dump-phash")]
        {
            use image_hasher::{
                FilterType,
                HashAlg,
                HasherConfig,
            };

            let mut output_image = image::ImageBuffer::new(image.width(), image.height());
            for (pixel, &idx) in output_image.pixels_mut().zip(&paletted_pixels) {
                let lab = palette[idx];
                let rgb: palette::Srgb = lab.into_color();
                let rgb = rgb.into_format::<u8>();
                *pixel = Rgb([rgb.red, rgb.green, rgb.blue]);
            }

            let hasher = HasherConfig::new()
                .hash_alg(HashAlg::DoubleGradient)
                .resize_filter(FilterType::Lanczos3)
                .hash_size(32, 32)
                .to_hasher();

            let hash_in = hasher.hash_image(image);
            let hash_out = hasher.hash_image(&output_image);

            println!("Hash Distance: {}", hash_in.dist(&hash_out));
        }

        #[cfg(feature = "dump-image")]
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
