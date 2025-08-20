//! A sixel library for encoding images.
//!
//! ## Basic Usage
//!
//! ### Simple Encoding
//!
//! ```rust
//! use a_sixel::BitMergeSixelEncoderBest;
//! use image::RgbaImage;
//!
//! let img = RgbaImage::new(100, 100);
//! println!("{}", <BitMergeSixelEncoderBest>::encode(img));
//! ```
//!
//! ### Loading and Encoding an Image File
//!
//! ```rust
//! use a_sixel::KMeansSixelEncoder;
//! use image;
//!
//! // Load an image from file
//! let image = image::open("examples/transparent.png").unwrap().to_rgba8();
//!
//! // Encode with default settings (256 colors, Sierra dithering)
//! let sixel_output = <KMeansSixelEncoder>::encode(image);
//! println!("{}", sixel_output);
//! ```
//!
//! ### Custom Palette Size and Dithering
//!
//! ```rust
//! use a_sixel::{
//!     BitSixelEncoder,
//!     dither::NoDither,
//! };
//!
//! let image = image::open("examples/transparent.png").unwrap().to_rgba8();
//!
//! // Use 16 colors with no dithering for faster encoding
//! let sixel_output = BitSixelEncoder::<NoDither>::encode_with_palette_size(image, 16);
//! println!("{}", sixel_output);
//! ```
//!
//! ## Transparency
//! By default, `a-sixel` handles transparency by setting any fully-transparent
//! pixels to all-bits-zero. This translates to a transparent pixel in most
//! sixel implementations, but some terminals may not support this.
//!
//! Sixel does not natively support partial transparency, but this library does
//! have some support for rendering images as if partial transparency was
//! supported. If the `partial-transparency` feature is enabled, `a-sixel` will
//! query the terminal and attempt to determine the background color. Partially
//! transparent pixels will then be blended with this background color before
//! encoding. Note that with this approach, changing the terminal background
//! color will not update partially transparent pixels to match. You will need
//! to re-encode the image if the background color changes.
//!
//!
//! ## Choosing an Encoder
//! - I want good quality:
//!   - Use `BitMergeSixelEncoderBest` or `KMeansSixelEncoder`.
//! - I'm time constrained:
//!   - Use `BitMergeSixelEncoderLow`, `BitSixelEncoder`, or
//!     `OctreeSixelEncoder`.
//! - I'm _really_ time constrained and can sacrifice a little quality:
//!   - Use `BitSixelEncoder<NoDither>`.
//!
//! For a more detailed breakdown, here's the encoders by average speed and
//! quality against the test images (speed figures will vary) at 256 colors with
//! Sierra dithering:
//!
//! | Algorithm        |   MSE |  DSSIM | PHash Distance | Mean ΔE | Max ΔE | ΔE >2.3 | ΔE >5.0 | Execution Time (ms) |
//! | :--------------- | ----: | -----: | -------------: | ------: | -----: | ------: | ------: | ------------------: |
//! | adu              | 15.04 | 0.0052 |           8.86 |    1.79 |  12.80 |   31.8% |    4.4% |                1448 |
//! | bit              | 35.82 | 0.0132 |          31.14 |    3.16 |  11.03 |   64.5% |   15.1% |                 468 |
//! | bit-merge-low    | 10.67 | 0.0038 |          13.97 |    1.95 |   9.98 |   32.4% |    2.2% |                 855 |
//! | bit-merge        | 10.37 | 0.0037 |          13.48 |    1.89 |  10.03 |   31.0% |    2.2% |                1034 |
//! | bit-merge-better | 10.30 | 0.0037 |          13.07 |    1.85 |  10.22 |   30.6% |    2.2% |                1301 |
//! | bit-merge-best   | 10.28 | 0.0037 |          13.59 |    1.83 |  10.20 |   30.5% |    2.2% |                1532 |
//! | focal            | 14.65 | 0.0056 |          20.10 |    2.30 |   9.17 |   45.3% |    3.3% |                2538 |
//! | k-means          | 10.07 | 0.0036 |          13.28 |    1.80 |  10.14 |   29.1% |    2.2% |                3175 |
//! | k-medians        | 17.22 | 0.0067 |          19.10 |    2.56 |   9.98 |   50.8% |    4.7% |                9088 |
//! | median-cut       | 19.64 | 0.0059 |          16.45 |    2.24 |  10.36 |   42.2% |    5.9% |                 740 |
//! | octree           | 54.48 | 0.0148 |          26.03 |    3.89 |  12.49 |   78.6% |   25.4% |                 754 |
//! | wu               | 17.89 | 0.0068 |          21.03 |    2.34 |  10.24 |   46.3% |    5.1% |                1984 |
//!
//! **Note:** Execution time _includes_ the time taken to compute error
//! statistics - this is non-trivial. For example, exclusive of error statistics
//! computation, bit-no-dither takes <100ms on average. Performance figures will
//! vary based on machine, etc. They are only useful for comparing algorithms
//! against each other within this dataset.
//!
//! Here's the encoders at 16 colors with Sierra dithering:
//!
//! | Algorithm        |    MSE |  DSSIM | PHash Distance | Mean ΔE | Max ΔE | ΔE >2.3 | ΔE >5.0 | Execution Time (ms) |
//! | :--------------- | -----: | -----: | -------------: | ------: | -----: | ------: | ------: | ------------------: |
//! | adu              | 118.90 | 0.0364 |          40.86 |    4.04 |  18.38 |   65.7% |   33.8% |                 357 |
//! | bit              | 178.47 | 0.0490 |          59.79 |    5.53 |  16.61 |   89.0% |   51.4% |                 325 |
//! | bit-merge-low    |  95.61 | 0.0302 |          41.59 |    3.97 |  16.26 |   67.4% |   31.4% |                 631 |
//! | bit-merge        |  94.53 | 0.0302 |          41.48 |    3.95 |  16.15 |   67.0% |   31.3% |                 800 |
//! | bit-merge-better |  96.11 | 0.0299 |          41.55 |    3.96 |  16.17 |   67.9% |   31.4% |                1078 |
//! | bit-merge-best   |  95.44 | 0.0297 |          41.69 |    3.96 |  16.46 |   67.0% |   31.4% |                1297 |
//! | focal            | 116.27 | 0.0350 |          48.48 |    4.35 |  16.87 |   71.7% |   34.1% |                2313 |
//! | k-means          |  99.36 | 0.0309 |          42.83 |    3.99 |  16.39 |   66.9% |   31.4% |                 702 |
//! | k-medians        | 173.95 | 0.0533 |          59.62 |    5.57 |  16.23 |   90.8% |   49.7% |                7255 |
//! | median-cut       | 164.52 | 0.0374 |          45.28 |    4.68 |  16.72 |   73.7% |   42.3% |                 395 |
//! | octree           | 459.37 | 0.0845 |          75.03 |    7.69 |  18.87 |   98.3% |   73.5% |                 477 |
//! | wu               | 125.84 | 0.0386 |          50.52 |    4.48 |  16.70 |   74.5% |   39.2% |                 929 |
#![cfg_attr(all(doc, ENABLE_DOC_AUTO_CFG), feature(doc_auto_cfg))]

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
#[cfg(feature = "wu")]
pub mod wu;

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
    RgbaImage,
};
use palette::{
    Hsl,
    IntoColor,
    Lab,
    encoding::Srgb,
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
pub use crate::bit::BitPaletteBuilder;
#[cfg(feature = "bit-merge")]
pub use crate::bitmerge::BitMergePaletteBuilder;
#[cfg(feature = "bit-merge")]
use crate::dither::{
    Dither,
    Sierra,
};
#[cfg(feature = "focal")]
pub use crate::focal::FocalPaletteBuilder;
#[cfg(feature = "k-means")]
pub use crate::kmeans::KMeansPaletteBuilder;
#[cfg(feature = "k-medians")]
pub use crate::kmedians::KMediansPaletteBuilder;
#[cfg(feature = "median-cut")]
pub use crate::median_cut::MedianCutPaletteBuilder;
#[cfg(feature = "octree")]
pub use crate::octree::OctreePaletteBuilder;
#[cfg(feature = "wu")]
pub use crate::wu::WuPaletteBuilder;

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
    const NAME: &'static str;

    /// Take in an image and return a quantized palette based on the colors in
    /// the image. The returned vector may be `<= palette_size` in length.
    fn build_palette(image: &RgbImage, palette_size: usize) -> Vec<Lab>;
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
/// [`ADUSixelEncoder`], which uses the [`ADUPaletteBuilder`].
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
pub struct SixelEncoder<P: PaletteBuilder = BitPaletteBuilder, D: Dither = Sierra> {
    _p: std::marker::PhantomData<P>,
    _d: std::marker::PhantomData<D>,
}

#[cfg(feature = "adu")]
pub type ADUSixelEncoder<D = Sierra> = SixelEncoder<ADUPaletteBuilder, D>;
#[cfg(feature = "bit-merge")]
pub type BitMergeSixelEncoderLow<D = Sierra> = SixelEncoder<BitMergePaletteBuilder<{ 1 << 14 }>, D>;
#[cfg(feature = "bit-merge")]
pub type BitMergeSixelEncoder<D = Sierra> = SixelEncoder<BitMergePaletteBuilder, D>;
#[cfg(feature = "bit-merge")]
pub type BitMergeSixelEncoderBetter<D = Sierra> =
    SixelEncoder<BitMergePaletteBuilder<{ 1 << 20 }>, D>;
#[cfg(feature = "bit-merge")]
pub type BitMergeSixelEncoderBest<D = Sierra> =
    SixelEncoder<BitMergePaletteBuilder<{ 1 << 21 }>, D>;
pub type BitSixelEncoder<D = Sierra> = SixelEncoder<BitPaletteBuilder, D>;
#[cfg(feature = "focal")]
pub type FocalSixelEncoder<D = Sierra> = SixelEncoder<FocalPaletteBuilder, D>;
#[cfg(feature = "k-means")]
pub type KMeansSixelEncoder<D = Sierra> = SixelEncoder<KMeansPaletteBuilder, D>;
#[cfg(feature = "k-medians")]
pub type KMediansSixelEncoder<D = Sierra> = SixelEncoder<KMediansPaletteBuilder, D>;
#[cfg(feature = "median-cut")]
pub type MedianCutSixelEncoder<D = Sierra> = SixelEncoder<MedianCutPaletteBuilder, D>;
#[cfg(feature = "octree")]
pub type OctreeSixelEncoder<D = Sierra, const USE_MIN_HEAP: bool = false> =
    SixelEncoder<OctreePaletteBuilder<USE_MIN_HEAP>, D>;
#[cfg(feature = "wu")]
pub type WuSixelEncoder<D = Sierra> = SixelEncoder<WuPaletteBuilder, D>;

impl<P: PaletteBuilder, D: Dither> SixelEncoder<P, D> {
    /// Encode an RGBA image into sixel format with the default palette size of
    /// 256 colors.
    ///
    /// This is a convenience method that calls
    /// [`encode_with_palette_size`](Self::encode_with_palette_size)
    /// with a palette size of 256.
    ///
    /// # Arguments
    ///
    /// * `rgba` - An RGBA image to encode. The alpha channel is used for
    ///   transparency handling.
    ///
    /// # Returns
    ///
    /// A `String` containing the sixel-encoded image data, ready to be printed
    /// to a sixel-capable terminal.
    ///
    /// # Transparency Handling
    ///
    /// - **Fully transparent pixels** (alpha = 0): Encoded as transparent sixel
    ///   pixels
    /// - **Partially transparent pixels**: If the `partial-transparency`
    ///   feature is enabled, these are blended with the detected terminal
    ///   background color. Otherwise, treated as opaque.
    /// - **Opaque pixels** (alpha = 255): Encoded normally
    pub fn encode(rgba: RgbaImage) -> String {
        Self::encode_with_palette_size(rgba, 256)
    }

    /// Encode an RGBA image into sixel format with a custom palette size.
    ///
    /// This method provides full control over the color quantization process by
    /// allowing you to specify the exact number of colors in the resulting
    /// palette. The palette size directly affects both the quality and size
    /// of the output.
    ///
    /// # Arguments
    ///
    /// * `rgba` - An RGBA image to encode. The alpha channel is used for
    ///   transparency handling.
    /// * `palette_size` - The number of colors to use in the palette. Valid
    ///   range is 1-256. Will be automatically clamped if the image has fewer
    ///   unique colors.
    ///
    /// # Returns
    ///
    /// A `String` containing the sixel-encoded image data, ready to be printed
    /// to a sixel-capable terminal.
    ///
    /// # Transparency Handling
    ///
    /// Same as [`encode`](Self::encode) - see that method's documentation for
    /// details.
    pub fn encode_with_palette_size(
        #[allow(unused_mut)] mut rgba: RgbaImage,
        palette_size: usize,
    ) -> String {
        #[cfg(feature = "partial-transparency")]
        {
            use std::time::Duration;

            let bg_color = termbg::rgb(Duration::from_millis(100))
                .map(|rgb| {
                    Rgb([
                        (rgb.r as f32 / u16::MAX as f32 * u8::MAX as f32) as u8,
                        (rgb.g as f32 / u16::MAX as f32 * u8::MAX as f32) as u8,
                        (rgb.b as f32 / u16::MAX as f32 * u8::MAX as f32) as u8,
                    ])
                })
                .unwrap_or(Rgb([0, 0, 0]));
            rgba.par_pixels_mut().for_each(|pixel| {
                use image::{
                    Pixel,
                    Rgba,
                };

                let mut color = Rgba([bg_color[0], bg_color[1], bg_color[2], pixel[3]]);
                color.blend(pixel);
                *pixel = color;
            });
        }
        let image = RgbImage::from_raw(
            rgba.width(),
            rgba.height(),
            rgba.pixels()
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let image = &image;
        let palette = if image.width().saturating_mul(image.height()) < palette_size as u32 {
            image.pixels().copied().map(rgb_to_lab).collect::<Vec<_>>()
        } else {
            P::build_palette(image, palette_size)
        };

        let mut sixel_string = r#"Pq"1;1;"#.to_string();
        sixel_string
            .write_fmt(format_args!("{};{}", image.height(), image.width()))
            .expect("Failed to write sixel bounds");

        if image.width() > 0 && image.height() > 0 {
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

            let paletted_pixels = D::dither_and_palettize(image, &palette);

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

                println!("MSE: {:.2} ({} colors)", mse, palette_size);
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
                let five_threshold = differences.iter().copied().filter(|d| *d > 5.0).count()
                    as f32
                    / (image.width() * image.height()) as f32;

                println!("Mean DeltaE: {:.2} ({} colors)", mean_diff, palette_size);
                println!("Max DeltaE: {:.2} ({} colors)", max_diff, palette_size);
                println!(
                    "DeltaE > 2.3: {:.2} ({} colors)",
                    two_three_threshold, palette_size
                );
                println!(
                    "DeltaE > 5.0: {:.2} ({} colors)",
                    five_threshold, palette_size
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

                println!("DSSIM: {:.4} ({} colors)", dssim, palette_size);
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

                println!(
                    "Hash Distance: {} ({} colors)",
                    hash_in.dist(&hash_out),
                    palette_size
                );
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

            let rgba_pixels = rgba.pixels().collect::<Vec<_>>();
            let rows = paletted_pixels
                .chunks(image.width() as usize)
                .zip(rgba_pixels.chunks(image.width() as usize))
                .collect::<Vec<_>>();

            let mut strings = vec![String::new(); rows.len().div_ceil(6)];
            rows.par_chunks(6)
                .zip(&mut strings)
                .for_each(|(stack, sixel_string)| {
                    let row_palette =
                        Vec::from_iter((0..palette_size).map(|_| AtomicBool::new(false)));
                    stack
                        .par_iter()
                        .flat_map(|(row, _)| row.par_iter().copied())
                        .for_each(|idx| {
                            row_palette[idx].store(true, Ordering::Relaxed);
                        });

                    for (color, _) in row_palette
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| v.load(Ordering::Relaxed))
                    {
                        let mut stack_string = SixelRow::new(sixel_string, color);
                        for idx in 0..stack[0].0.len() {
                            let bit0 = (stack[0].0[idx] == color && stack[0].1[idx][3] != 0) as u8;
                            let bit1 = (stack
                                .get(1)
                                .filter(|(_, v)| v[idx][3] != 0)
                                .map(|(r, _)| r[idx])
                                == Some(color)) as u8;
                            let bit2 = (stack
                                .get(2)
                                .filter(|(_, v)| v[idx][3] != 0)
                                .map(|(r, _)| r[idx])
                                == Some(color)) as u8;
                            let bit3 = (stack
                                .get(3)
                                .filter(|(_, v)| v[idx][3] != 0)
                                .map(|(r, _)| r[idx])
                                == Some(color)) as u8;
                            let bit4 = (stack
                                .get(4)
                                .filter(|(_, v)| v[idx][3] != 0)
                                .map(|(r, _)| r[idx])
                                == Some(color)) as u8;
                            let bit5 = (stack
                                .get(5)
                                .filter(|(_, v)| v[idx][3] != 0)
                                .map(|(r, _)| r[idx])
                                == Some(color)) as u8;

                            let bits = bit0
                                | (bit1 << 1)
                                | (bit2 << 2)
                                | (bit3 << 3)
                                | (bit4 << 4)
                                | (bit5 << 5);
                            let char = num2six(bits);
                            stack_string.push(char);
                        }
                        stack_string.finalize();
                    }
                    sixel_string.push('-');
                });

            sixel_string.extend(strings);
        }

        sixel_string.push_str(r#"\"#);
        sixel_string
    }
}
fn rgb_to_lab(Rgb([r, g, b]): Rgb<u8>) -> Lab {
    palette::rgb::Rgb::<Srgb, _>::new(r, g, b)
        .into_format::<f32>()
        .into_color()
}
