use std::fs::read;

#[cfg(feature = "adu")]
use a_sixel::ADUSixelEncoder;
#[cfg(feature = "focal")]
use a_sixel::FocalSixelEncoder;
#[cfg(feature = "k-means")]
use a_sixel::KMeansSixelEncoder;
#[cfg(feature = "k-medians")]
use a_sixel::KMediansSixelEncoder;
#[cfg(feature = "median-cut")]
use a_sixel::MedianCutSixelEncoder;
#[cfg(feature = "octree")]
use a_sixel::OctreeSixelEncoder;
#[cfg(feature = "wu")]
use a_sixel::WuSixelEncoder;
#[cfg(feature = "bit-merge")]
use a_sixel::{
    BitMergeSixelEncoder,
    BitMergeSixelEncoderBest,
    BitMergeSixelEncoderBetter,
    BitMergeSixelEncoderLow,
};
use a_sixel::{
    BitSixelEncoder,
    dither::{
        Bayer,
        NoDither,
        Sobol,
    },
};
use clap::Parser;
use strum::{
    Display,
    EnumString,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumString, Display)]
#[strum(ascii_case_insensitive, serialize_all = "kebab-case")]
enum PaletteFormat {
    #[cfg(feature = "adu")]
    Adu,
    Bit,
    #[cfg(feature = "bit-merge")]
    BitMergeLow,
    #[cfg(feature = "bit-merge")]
    BitMerge,
    #[cfg(feature = "bit-merge")]
    BitMergeBetter,
    #[cfg(feature = "bit-merge")]
    BitMergeBest,
    #[cfg(feature = "focal")]
    Focal,
    #[cfg(feature = "k-means")]
    #[strum(serialize = "kmeans", serialize = "k-means")]
    KMeans,
    #[cfg(feature = "k-medians")]
    #[strum(
        serialize = "kmedians",
        serialize = "kmed",
        serialize = "k-medians",
        serialize = "k-med"
    )]
    KMedians,
    #[cfg(feature = "median-cut")]
    MedianCut,
    #[cfg(feature = "octree")]
    Octree,
    #[cfg(feature = "wu")]
    Wu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumString, Display)]
#[strum(ascii_case_insensitive, serialize_all = "kebab-case")]
enum Dither {
    #[strum(serialize = "no", serialize = "no-dither")]
    No,
    Sierra,
    Sobol,
    Bayer,
}

#[derive(Debug, Parser)]
struct Args {
    /// The path to the image file to be loaded and rendered
    #[clap(long, short)]
    image_path: String,

    /// The palette size to use for quantization, will be rounded to the nearest
    /// power of 2.
    #[clap(long, short, default_value_t = 256)]
    palette_size: usize,

    /// The palette generator to use.
    #[clap(long, short = 'f', default_value_t = PaletteFormat::Bit)]
    palette_format: PaletteFormat,

    /// Whether to use Sobol dithering instead of the default dithering.
    #[clap(long, short, default_value_t = Dither::Sierra)]
    dither: Dither,

    /// Whether to display the image in a terminal that supports sixel graphics.
    #[clap(long, short = 's', default_value_t = false)]
    show: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let image_path = args.image_path;

    println!(
        "Loading image: {} ({} colors)",
        std::path::Path::new(&image_path)
            .file_stem()
            .unwrap()
            .display(),
        args.palette_size
    );
    println!(
        "algorithm: {} ({} colors)",
        args.palette_format, args.palette_size
    );

    let timer = std::time::Instant::now();

    let image = image::load_from_memory(&read(image_path)?)?;
    let image = image.to_rgba8();

    let six = match args.palette_format {
        #[cfg(feature = "adu")]
        PaletteFormat::Adu => match args.dither {
            Dither::No => {
                <ADUSixelEncoder<NoDither>>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Sobol => {
                <ADUSixelEncoder<Sobol>>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Sierra => <ADUSixelEncoder>::encode_with_palette_size(image, args.palette_size),
            Dither::Bayer => {
                <ADUSixelEncoder<Bayer>>::encode_with_palette_size(image, args.palette_size)
            }
        },
        PaletteFormat::Bit => match args.dither {
            Dither::No => {
                <BitSixelEncoder<NoDither>>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Sobol => {
                <BitSixelEncoder<Sobol>>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Sierra => <BitSixelEncoder>::encode_with_palette_size(image, args.palette_size),
            Dither::Bayer => {
                <BitSixelEncoder<Bayer>>::encode_with_palette_size(image, args.palette_size)
            }
        },
        #[cfg(feature = "bit-merge")]
        PaletteFormat::BitMergeLow => match args.dither {
            Dither::No => <BitMergeSixelEncoderLow<NoDither>>::encode_with_palette_size(
                image,
                args.palette_size,
            ),
            Dither::Sobol => {
                <BitMergeSixelEncoderLow<Sobol>>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Sierra => {
                <BitMergeSixelEncoderLow>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Bayer => {
                <BitMergeSixelEncoderLow<Bayer>>::encode_with_palette_size(image, args.palette_size)
            }
        },
        #[cfg(feature = "bit-merge")]
        PaletteFormat::BitMerge => match args.dither {
            Dither::No => {
                <BitMergeSixelEncoder<NoDither>>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Sobol => {
                <BitMergeSixelEncoder<Sobol>>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Sierra => {
                <BitMergeSixelEncoder>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Bayer => {
                <BitMergeSixelEncoder<Bayer>>::encode_with_palette_size(image, args.palette_size)
            }
        },
        #[cfg(feature = "bit-merge")]
        PaletteFormat::BitMergeBetter => match args.dither {
            Dither::No => <BitMergeSixelEncoderBetter<NoDither>>::encode_with_palette_size(
                image,
                args.palette_size,
            ),
            Dither::Sobol => <BitMergeSixelEncoderBetter<Sobol>>::encode_with_palette_size(
                image,
                args.palette_size,
            ),
            Dither::Sierra => {
                <BitMergeSixelEncoderBetter>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Bayer => <BitMergeSixelEncoderBetter<Bayer>>::encode_with_palette_size(
                image,
                args.palette_size,
            ),
        },
        #[cfg(feature = "bit-merge")]
        PaletteFormat::BitMergeBest => match args.dither {
            Dither::No => <BitMergeSixelEncoderBest<NoDither>>::encode_with_palette_size(
                image,
                args.palette_size,
            ),
            Dither::Sobol => <BitMergeSixelEncoderBest<Sobol>>::encode_with_palette_size(
                image,
                args.palette_size,
            ),
            Dither::Sierra => {
                <BitMergeSixelEncoderBest>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Bayer => <BitMergeSixelEncoderBest<Bayer>>::encode_with_palette_size(
                image,
                args.palette_size,
            ),
        },
        #[cfg(feature = "focal")]
        PaletteFormat::Focal => match args.dither {
            Dither::No => {
                <FocalSixelEncoder<NoDither>>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Sobol => {
                <FocalSixelEncoder<Sobol>>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Sierra => {
                <FocalSixelEncoder>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Bayer => {
                <FocalSixelEncoder<Bayer>>::encode_with_palette_size(image, args.palette_size)
            }
        },
        #[cfg(feature = "k-means")]
        PaletteFormat::KMeans => match args.dither {
            Dither::No => {
                <KMeansSixelEncoder<NoDither>>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Sobol => {
                <KMeansSixelEncoder<Sobol>>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Sierra => {
                <KMeansSixelEncoder>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Bayer => {
                <KMeansSixelEncoder<Bayer>>::encode_with_palette_size(image, args.palette_size)
            }
        },
        #[cfg(feature = "k-medians")]
        PaletteFormat::KMedians => match args.dither {
            Dither::No => {
                <KMediansSixelEncoder<NoDither>>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Sobol => {
                <KMediansSixelEncoder<Sobol>>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Sierra => {
                <KMediansSixelEncoder>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Bayer => {
                <KMediansSixelEncoder<Bayer>>::encode_with_palette_size(image, args.palette_size)
            }
        },
        #[cfg(feature = "median-cut")]
        PaletteFormat::MedianCut => match args.dither {
            Dither::No => <MedianCutSixelEncoder<NoDither>>::encode_with_palette_size(
                image,
                args.palette_size,
            ),
            Dither::Sobol => {
                <MedianCutSixelEncoder<Sobol>>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Sierra => {
                <MedianCutSixelEncoder>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Bayer => {
                <MedianCutSixelEncoder<Bayer>>::encode_with_palette_size(image, args.palette_size)
            }
        },
        #[cfg(feature = "octree")]
        PaletteFormat::Octree => match args.dither {
            Dither::No => {
                <OctreeSixelEncoder<NoDither>>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Sobol => {
                <OctreeSixelEncoder<Sobol>>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Sierra => {
                <OctreeSixelEncoder>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Bayer => {
                <OctreeSixelEncoder<Bayer>>::encode_with_palette_size(image, args.palette_size)
            }
        },
        #[cfg(feature = "wu")]
        PaletteFormat::Wu => match args.dither {
            Dither::No => {
                <WuSixelEncoder<NoDither>>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Sobol => {
                <WuSixelEncoder<Sobol>>::encode_with_palette_size(image, args.palette_size)
            }
            Dither::Sierra => <WuSixelEncoder>::encode_with_palette_size(image, args.palette_size),
            Dither::Bayer => {
                <WuSixelEncoder<Bayer>>::encode_with_palette_size(image, args.palette_size)
            }
        },
    };

    println!(
        "Time taken: {}ms ({} colors)",
        timer.elapsed().as_millis(),
        args.palette_size
    );

    if args.show {
        println!("{six}");
    }

    Ok(())
}
