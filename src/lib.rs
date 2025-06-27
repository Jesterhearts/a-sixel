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
//!   - Use [`BitMergeSixelEncoderLow`], or [`BitSixelEncoder`].
//! - I'm _really_ time constrained and can sacrifice a little quality:
//!   - Use [`BitSixelEncoder<NoDither>`].
//!
//! For a more detailed breakdown, here's the encoders by average speed and
//! quality against the test images (speed figures will vary) at 256 colors with
//! Sierra dithering:
//!
//! | Algorithm        |   MSE | DSSIM  | PHash Distance | Mean ΔE | Max ΔE | ΔE >2.3 | ΔE >5.0 | Execution Time (ms) |
//! | :--------------- | ----: | :----: | -------------: | ------: | -----: | ------: | ------: | ------------------: |
//! | adu              | 15.07 | 0.0053 |           8.66 |    1.79 |  12.88 |   31.6% |    4.4% |                1416 |
//! | bit              | 35.82 | 0.0132 |          31.14 |    3.16 |  11.03 |   64.5% |   15.1% |                 426 |
//! | bit-no-dither    | 31.78 | 0.0214 |          39.10 |    3.09 |  10.23 |   64.0% |   13.4% |                 274 |
//! | bit-merge-low    | 10.67 | 0.0038 |          13.97 |    1.95 |   9.98 |   32.4% |    2.2% |                 785 |
//! | bit-merge        | 10.36 | 0.0037 |          13.55 |    1.89 |  10.01 |   31.0% |    2.2% |                 932 |
//! | bit-merge-better | 10.31 | 0.0037 |          13.45 |    1.85 |  10.21 |   30.6% |    2.2% |                1275 |
//! | bit-merge-best   | 10.29 | 0.0037 |          13.52 |    1.83 |  10.26 |   30.6% |    2.2% |                1496 |
//! | focal            | 14.62 | 0.0056 |          19.97 |    2.30 |   9.16 |   45.3% |    3.3% |                2428 |
//! | k-means          | 10.07 | 0.0036 |          13.07 |    1.80 |  10.17 |   29.1% |    2.2% |                2996 |
//! | k-medians        | 17.67 | 0.0068 |          21.07 |    2.61 |  10.17 |   53.6% |    5.1% |                7305 |
//! | median-cut       | 19.63 | 0.0059 |          16.45 |    2.24 |  10.36 |   42.2% |    5.9% |                 692 |
//! | octree           | 54.48 | 0.0148 |          26.03 |    3.89 |  12.49 |   78.6% |   25.4% |                 682 |
//! | wu               | 17.89 | 0.0068 |          21.03 |    2.34 |  10.24 |   46.3% |    5.1% |                1853 |
//!
//! **Note:** Execution time _includes_ the time taken to compute error
//! statistics - this is non-trivial. For example, exclusive of error statistics
//! computation, bit-no-dither takes <100ms on average. Performance figures will
//! vary based on machine, etc. They are only useful for comparing algorithms
//! against each other within this dataset.
//!
//! Here's the encoders at 16 colors with Sierra dithering:
//!
//! | Algorithm  |    MSE | DSSIM | PHash Distance | Mean ΔE | Max ΔE | ΔE >2.3 | ΔE >5.0 | Execution Time (ms) |
//! | :--------- | -----: | :---: | -------------: | ------: | -----: | ------: | ------: | ------------------: |
//! | adu        | 116.85 | 0.036 |          39.83 |    4.02 |  18.39 |     66% |     33% |                 332 |
//! | bit        | 178.47 | 0.049 |          59.79 |    5.53 |  16.61 |     89% |     51% |                 307 |
//! | bit-merge  |  95.17 | 0.030 |          41.52 |    3.95 |  16.16 |     67% |     31% |                 712 |
//! | focal      | 118.57 | 0.035 |          48.59 |    4.36 |  16.88 |     72% |     34% |                2150 |
//! | k-means    |  99.36 | 0.031 |          43.10 |    3.99 |  16.41 |     67% |     31% |                 637 |
//! | k-medians  | 166.88 | 0.050 |          60.59 |    5.48 |  16.77 |     88% |     52% |                5447 |
//! | median-cut | 164.52 | 0.037 |          45.28 |    4.68 |  16.72 |     74% |     42% |                 374 |
//! | octree     | 459.37 | 0.085 |          75.07 |    7.69 |  18.89 |     98% |     74% |                 446 |
//! | wu         | 125.84 | 0.039 |          50.52 |    4.48 |  16.70 |     75% |     39% |                 906 |

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
#[cfg(feature = "wu")]
pub use crate::wu::WuPaletteBuilder;
#[cfg(feature = "wu")]
use crate::wu::WuSixelEncoder256;
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
#[cfg(feature = "wu")]
pub type WuSixelEncoder<D = Sierra> = WuSixelEncoder256<D>;

impl<P: PaletteBuilder, D: Dither> SixelEncoder<P, D> {
    pub fn encode(#[allow(unused_mut)] mut rgba: RgbaImage) -> String {
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
        let palette = if image.width().saturating_mul(image.height()) < P::PALETTE_SIZE as u32 {
            image.pixels().copied().map(rgb_to_lab).collect::<Vec<_>>()
        } else {
            P::build_palette(image)
        };

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
                    Vec::from_iter((0..P::PALETTE_SIZE).map(|_| AtomicBool::new(false)));
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
                        let bits = if stack[0].1[idx][3] == 0 {
                            0
                        } else {
                            (stack[0].0[idx] == color) as u8
                                | ((stack.get(1).map(|(r, _)| r[idx]) == Some(color)) as u8) << 1
                                | ((stack.get(2).map(|(r, _)| r[idx]) == Some(color)) as u8) << 2
                                | ((stack.get(3).map(|(r, _)| r[idx]) == Some(color)) as u8) << 3
                                | ((stack.get(4).map(|(r, _)| r[idx]) == Some(color)) as u8) << 4
                                | ((stack.get(5).map(|(r, _)| r[idx]) == Some(color)) as u8) << 5
                        };
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
