//! A sixel library for encoding images.
//!
//! ## Basic Usage
//!
//! ### Simple Encoding
//!
//! ```rust
//! use a_sixel::PaletteBuilder;
//! use a_sixel::SixelEncoder;
//! use a_sixel::dither::Dither;
//! use image::RgbaImage;
//!
//! let img = RgbaImage::new(100, 100);
//! println!(
//!     "{}",
//!     SixelEncoder::new(PaletteBuilder::BitMergeBest, Dither::Sierra).encode(img)
//! );
//! ```
//!
//! ### Loading and Encoding an Image File
//!
//! ```rust
//! use a_sixel::PaletteBuilder;
//! use a_sixel::SixelEncoder;
//! use a_sixel::dither::Dither;
//!
//! // Load an image from file
//! let image = image::open("examples/transparent.png").unwrap().to_rgba8();
//!
//! // Encode with 256 colors and Sierra dithering
//! let encoder = SixelEncoder::new(PaletteBuilder::KMeans, Dither::Sierra);
//! let sixel_output = encoder.encode(image);
//! println!("{}", sixel_output);
//! ```
//!
//! ### Custom Palette Size and Dithering
//!
//! ```rust
//! use a_sixel::PaletteBuilder;
//! use a_sixel::SixelEncoder;
//! use a_sixel::dither::Dither;
//!
//! let image = image::open("examples/transparent.png").unwrap().to_rgba8();
//!
//! // Use 16 colors with no dithering for faster encoding
//! let encoder = SixelEncoder::new(PaletteBuilder::Bit, Dither::None);
//! let sixel_output = encoder.encode_with_palette_size(image, 16);
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
//! ## Choosing a `PaletteBuilder`
//! - I want good quality:
//!   - Use [`PaletteBuilder::BitMergeBest`] or [`PaletteBuilder::KMeans`].
//! - I'm time constrained:
//!   - Use [`PaletteBuilder::BitMergeLow`], [`PaletteBuilder::Bit`], or
//!     [`PaletteBuilder::Octree`].
//! - I'm _really_ time constrained and can sacrifice a little quality:
//!   - Use [`PaletteBuilder::Bit`] with [`Dither::None`](dither::Dither::None).
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
//! | focal            | 31.10 | 0.0091 |          19.72 |    3.34 |   8.41 |   73.9% |   13.1% |                 821 |
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
//! | focal            | 345.08 | 0.0585 |          56.66 |    7.23 |  16.65 |   95.6% |   74.8% |                 433 |
//! | k-means          |  99.36 | 0.0309 |          42.83 |    3.99 |  16.39 |   66.9% |   31.4% |                 702 |
//! | k-medians        | 173.95 | 0.0533 |          59.62 |    5.57 |  16.23 |   90.8% |   49.7% |                7255 |
//! | median-cut       | 164.52 | 0.0374 |          45.28 |    4.68 |  16.72 |   73.7% |   42.3% |                 395 |
//! | octree           | 459.37 | 0.0845 |          75.03 |    7.69 |  18.87 |   98.3% |   73.5% |                 477 |
//! | wu               | 125.84 | 0.0386 |          50.52 |    4.48 |  16.70 |   74.5% |   39.2% |                 929 |
#![cfg_attr(all(doc, ENABLE_DOC_AUTO_CFG), feature(doc_cfg))]

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

use image::Rgba;
use image::RgbaImage;
use palette::Hsl;
use palette::IntoColor;
use palette::Lab;
use palette::encoding::Srgb;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;
#[cfg(feature = "strum")]
use strum::Display;
#[cfg(feature = "strum")]
use strum::EnumString;

/// The palette building algorithm to use for color quantization.
///
/// # Choosing an algorithm
/// - [`PaletteBuilder::BitMergeBest`] or [`PaletteBuilder::KMeans`] for best
///   quality.
/// - [`PaletteBuilder::BitMergeLow`], [`PaletteBuilder::Bit`], or
///   [`PaletteBuilder::Octree`] for speed.
/// - [`PaletteBuilder::Bit`] combined with
///   [`Dither::None`](dither::Dither::None) for maximum speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "strum", derive(EnumString, Display))]
#[cfg_attr(
    feature = "strum",
    strum(ascii_case_insensitive, serialize_all = "kebab-case")
)]
#[non_exhaustive]
pub enum PaletteBuilder {
    /// Adaptive Distributive Units algorithm.
    #[cfg(feature = "adu")]
    Adu,
    /// Fast bit-dilation bucketing.
    Bit,
    /// Bit bucketing + k-means + agglomerative merge (low quality preset).
    #[cfg(feature = "bit-merge")]
    BitMergeLow,
    /// Bit bucketing + k-means + agglomerative merge (default preset).
    #[cfg(feature = "bit-merge")]
    BitMerge,
    /// Bit bucketing + k-means + agglomerative merge (better quality preset).
    #[cfg(feature = "bit-merge")]
    BitMergeBetter,
    /// Bit bucketing + k-means + agglomerative merge (best quality preset).
    #[cfg(feature = "bit-merge")]
    BitMergeBest,
    /// Spectral-residual peak isolation algorithm.
    #[cfg(feature = "focal")]
    Focal,
    /// K-means clustering.
    #[cfg(feature = "k-means")]
    KMeans,
    /// K-medians clustering.
    #[cfg(feature = "k-medians")]
    KMedians,
    /// Median cut quantization.
    #[cfg(feature = "median-cut")]
    MedianCut,
    /// Octree quantization (max-heap merging).
    #[cfg(feature = "octree")]
    Octree,
    /// Octree quantization (min-heap merging).
    #[cfg(feature = "octree")]
    OctreeMinHeap,
    /// Wu's PCA-based quantization.
    #[cfg(feature = "wu")]
    Wu,
}

impl PaletteBuilder {
    #[cfg(feature = "dump-image")]
    fn name(&self) -> &'static str {
        match self {
            #[cfg(feature = "adu")]
            Self::Adu => "ADU",
            Self::Bit => "Bit",
            #[cfg(feature = "bit-merge")]
            Self::BitMergeLow | Self::BitMerge | Self::BitMergeBetter | Self::BitMergeBest => {
                "Bit-Merge"
            }
            #[cfg(feature = "focal")]
            Self::Focal => "Focal",
            #[cfg(feature = "k-means")]
            Self::KMeans => "K-Means",
            #[cfg(feature = "k-medians")]
            Self::KMedians => "K-Medians",
            #[cfg(feature = "median-cut")]
            Self::MedianCut => "Median-Cut",
            #[cfg(feature = "octree")]
            Self::Octree | Self::OctreeMinHeap => "Octree",
            #[cfg(feature = "wu")]
            Self::Wu => "Wu",
        }
    }

    fn build_palette(
        &self,
        image: &RgbaImage,
        palette_size: usize,
    ) -> Vec<Lab> {
        match self {
            #[cfg(feature = "adu")]
            Self::Adu => adu::ADUPaletteBuilder::build_palette(image, palette_size),
            Self::Bit => bit::BitPaletteBuilder::build_palette(image, palette_size),
            #[cfg(feature = "bit-merge")]
            Self::BitMergeLow => {
                bitmerge::BitMergePaletteBuilder::<{ 1 << 14 }>::build_palette(image, palette_size)
            }
            #[cfg(feature = "bit-merge")]
            Self::BitMerge => {
                <bitmerge::BitMergePaletteBuilder>::build_palette(image, palette_size)
            }
            #[cfg(feature = "bit-merge")]
            Self::BitMergeBetter => {
                bitmerge::BitMergePaletteBuilder::<{ 1 << 20 }>::build_palette(image, palette_size)
            }
            #[cfg(feature = "bit-merge")]
            Self::BitMergeBest => {
                bitmerge::BitMergePaletteBuilder::<{ 1 << 21 }>::build_palette(image, palette_size)
            }
            #[cfg(feature = "focal")]
            Self::Focal => focal::FocalPaletteBuilder::build_palette(image, palette_size),
            #[cfg(feature = "k-means")]
            Self::KMeans => kmeans::KMeansPaletteBuilder::build_palette(image, palette_size),
            #[cfg(feature = "k-medians")]
            Self::KMedians => kmedians::KMediansPaletteBuilder::build_palette(image, palette_size),
            #[cfg(feature = "median-cut")]
            Self::MedianCut => {
                median_cut::MedianCutPaletteBuilder::build_palette(image, palette_size)
            }
            #[cfg(feature = "octree")]
            Self::Octree => {
                octree::OctreePaletteBuilder::<false>::build_palette(image, palette_size)
            }
            #[cfg(feature = "octree")]
            Self::OctreeMinHeap => {
                octree::OctreePaletteBuilder::<true>::build_palette(image, palette_size)
            }
            #[cfg(feature = "wu")]
            Self::Wu => wu::WuPaletteBuilder::build_palette(image, palette_size),
        }
    }

    fn build_bucketer(
        &self,
        palette: &[Lab],
        palette_size: usize,
    ) -> dither::PaletteBucketer {
        match self {
            Self::Bit => dither::PaletteBucketer::Bit(bit::BitPaletteBuilder::build_bucketer(
                palette,
                palette_size,
            )),
            _ => dither::PaletteBucketer::KdTree(dither::KdTreeBucketer::new(palette)),
        }
    }
}

/// The main type for performing sixel encoding.
///
/// Combines a [`PaletteBuilder`] to generate a color palette from the input
/// image (sixel only supports up to 256 colors) with a
/// [`Dither`](dither::Dither) algorithm to apply dithering to the reduced color
/// image before encoding it into sixel format.
///
/// # Choosing a `PaletteBuilder`
/// - [`PaletteBuilder::BitMergeBest`] or [`PaletteBuilder::KMeans`] are good
///   default choices for minimizing the error across the image.
/// - [`PaletteBuilder::Focal`] is a good choice if the image has highlights and
///   other details that other encoders might squash, but is experimental. It is
///   a weighted k-means implementation. Depending on the image, `KMeans` may be
///   able to capture these highlights already, but it's worth trying if you're
///   trying to preserve specific image characteristics.
///
/// # Choosing a `Dither`
/// - [`Dither::Sierra`](dither::Dither::Sierra) is a good default choice for
///   dithering, as it produces high-quality results with minimal artifacts.
/// - [`Dither::None`](dither::Dither::None) can be used if performance is a
///   concern.
#[derive(Debug, Clone, Copy)]
pub struct SixelEncoder {
    pub algorithm: PaletteBuilder,
    pub dither: dither::Dither,
}

impl Default for SixelEncoder {
    fn default() -> Self {
        Self {
            algorithm: PaletteBuilder::Bit,
            dither: dither::Dither::Sierra,
        }
    }
}

impl SixelEncoder {
    pub fn new(
        algorithm: PaletteBuilder,
        dither: dither::Dither,
    ) -> Self {
        Self { algorithm, dither }
    }

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
    pub fn encode(
        &self,
        rgba: RgbaImage,
    ) -> String {
        self.encode_with_palette_size(rgba, 256)
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
        &self,
        #[allow(unused_mut)] mut image: RgbaImage,
        palette_size: usize,
    ) -> String {
        #[cfg(feature = "partial-transparency")]
        {
            use std::sync::LazyLock;
            use std::time::Duration;

            static BG_COLOR: LazyLock<Rgba<u8>> = LazyLock::new(|| {
                termbg::rgb(Duration::from_millis(100))
                    .map(|rgb| {
                        Rgba([
                            (rgb.r as f32 / u16::MAX as f32 * u8::MAX as f32) as u8,
                            (rgb.g as f32 / u16::MAX as f32 * u8::MAX as f32) as u8,
                            (rgb.b as f32 / u16::MAX as f32 * u8::MAX as f32) as u8,
                            u8::MAX,
                        ])
                    })
                    .unwrap_or(Rgba([0, 0, 0, u8::MAX]))
            });

            image.par_pixels_mut().for_each(|pixel| {
                use image::Pixel;
                use image::Rgba;

                let mut color = Rgba([BG_COLOR[0], BG_COLOR[1], BG_COLOR[2], pixel[3]]);
                color.blend(pixel);
                *pixel = color;
            });
        }
        let palette = if image.width().saturating_mul(image.height()) < palette_size as u32 {
            image.pixels().copied().map(rgba_to_lab).collect::<Vec<_>>()
        } else {
            self.algorithm.build_palette(&image, palette_size)
        };

        let mut sixel_string = r#"P9;1q"1;1;"#.to_string();
        push_usize(&mut sixel_string, image.width() as usize);
        sixel_string.push(';');
        push_usize(&mut sixel_string, image.height() as usize);

        if image.width() > 0 && image.height() > 0 {
            for (i, lab) in palette.iter().copied().enumerate() {
                let hsl: Hsl = lab.into_color();
                // Sixel hue is offset by 120 degrees from the common hue values.
                let deg = (hsl.hue.into_positive_degrees().round() as u16 + 120) % 360;

                sixel_string.push('#');
                push_usize(&mut sixel_string, i);
                sixel_string.push_str(";1;");
                push_usize(&mut sixel_string, deg as usize);
                sixel_string.push(';');
                push_usize(&mut sixel_string, (hsl.lightness * 100.0).round() as usize);
                sixel_string.push(';');
                push_usize(&mut sixel_string, (hsl.saturation * 100.0).round() as usize);
            }

            let bucketer = self.algorithm.build_bucketer(&palette, palette_size);
            let paletted_pixels = self
                .dither
                .dither_and_palettize(&image, &palette, &bucketer);

            #[cfg(feature = "dump-mse")]
            {
                use rayon::iter::IntoParallelRefIterator;

                let dequant = paletted_pixels
                    .iter()
                    .map(|&idx| palette[idx])
                    .collect::<Vec<_>>();
                let mse = dequant
                    .par_iter()
                    .zip(image.par_pixels())
                    .map(|(l, rgb)| {
                        use palette::color_difference::EuclideanDistance;
                        let lab = rgba_to_lab(*rgb);
                        lab.distance_squared(*l)
                    })
                    .sum::<f32>()
                    / (image.width() * image.height()) as f32;

                println!("MSE: {:.2} ({} colors)", mse, palette_size);
            }

            #[cfg(feature = "dump-delta-e")]
            {
                use rayon::iter::IntoParallelRefIterator;

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
                        let lab_rgb = rgba_to_lab(rgb);
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
                    .map(|Rgba([r, g, b, _])| rgb::RGB::new(r, g, b))
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
                use image_hasher::FilterType;
                use image_hasher::HashAlg;
                use image_hasher::HasherConfig;

                let mut output_image = image::ImageBuffer::new(image.width(), image.height());
                for (pixel, &idx) in output_image.pixels_mut().zip(&paletted_pixels) {
                    let lab = palette[idx];
                    let rgb: palette::Srgb = lab.into_color();
                    let rgb = rgb.into_format::<u8>();
                    *pixel = Rgba([rgb.red, rgb.green, rgb.blue, u8::MAX]);
                }

                let hasher = HasherConfig::new()
                    .hash_alg(HashAlg::DoubleGradient)
                    .resize_filter(FilterType::Lanczos3)
                    .hash_size(32, 32)
                    .to_hasher();

                let hash_in = hasher.hash_image(&image);
                let hash_out = hasher.hash_image(&output_image);

                println!(
                    "Hash Distance: {} ({} colors)",
                    hash_in.dist(&hash_out),
                    palette_size
                );
            }

            #[cfg(feature = "dump-image")]
            {
                use std::hash::BuildHasher;
                use std::hash::Hasher;
                use std::hash::RandomState;

                let mut output_image = image::ImageBuffer::new(image.width(), image.height());
                for (pixel, &idx) in output_image.pixels_mut().zip(&paletted_pixels) {
                    let lab = palette[idx];
                    let rgb: palette::Srgb = lab.into_color();
                    let rgb = rgb.into_format::<u8>();
                    *pixel = Rgba([rgb.red, rgb.green, rgb.blue, u8::MAX]);
                }
                let rand = BuildHasher::build_hasher(&RandomState::new()).finish();

                output_image
                    .save(format!("{}-{rand}.png", self.algorithm.name()))
                    .expect("Failed to save output image");
            }

            let width = image.width() as usize;

            let num_chunks = (paletted_pixels.len() / width).div_ceil(6);
            let chunk_capacity = width * 7;
            let mut strings =
                Vec::from_iter((0..num_chunks).map(|_| String::with_capacity(chunk_capacity)));

            paletted_pixels
                .par_chunks(width * 6)
                .zip(image.into_raw().par_chunks(width * 6 * 4))
                .zip(&mut strings)
                .for_each(|((palette_chunk, rgba_chunk), sixel_string)| {
                    let mut color_bits = vec![0u8; palette_size * width];
                    let mut color_used = vec![false; palette_size];
                    let chunk_height = palette_chunk.len() / width;

                    for row in 0..chunk_height {
                        let bit = 1u8 << row;
                        let row_offset = row * width;
                        for col in 0..width {
                            let pixel_idx = row_offset + col;
                            if rgba_chunk[pixel_idx * 4 + 3] != 0 {
                                let color = palette_chunk[pixel_idx];
                                color_bits[color * width + col] |= bit;
                                color_used[color] = true;
                            }
                        }
                    }

                    color_bits.par_iter_mut().for_each(|d| {
                        *d += 0x3f;
                    });
                    // SAFETY: 0x3f..=0x7e are valid ASCII bytes
                    let color_bits = unsafe { String::from_utf8_unchecked(color_bits) };

                    for (color, _) in color_used.iter().enumerate().filter(|(_, u)| **u) {
                        sixel_string.push('#');
                        push_usize(sixel_string, color);
                        let base = color * width;
                        sixel_string.push_str(&color_bits[base..base + width]);
                        sixel_string.push('$');
                    }
                    sixel_string.push('-');
                });

            sixel_string.extend(strings);
        }

        sixel_string.push_str(r#"\"#);
        sixel_string
    }
}

fn rgba_to_lab(Rgba([r, g, b, _]): Rgba<u8>) -> Lab {
    palette::rgb::Rgb::<Srgb, _>::new(r, g, b)
        .into_format::<f32>()
        .into_color()
}

fn push_usize(
    s: &mut String,
    n: usize,
) {
    s.push_str(itoa::Buffer::new().format(n));
}
