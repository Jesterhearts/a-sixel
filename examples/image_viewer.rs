use std::fs::read;

#[cfg(feature = "adu")]
use a_sixel::adu::{
    ADUSixelEncoder128,
    ADUSixelEncoder16,
    ADUSixelEncoder256,
    ADUSixelEncoder32,
    ADUSixelEncoder4,
    ADUSixelEncoder64,
    ADUSixelEncoder8,
    ADUSixelEncoderMono,
};
#[cfg(feature = "bit-merge")]
use a_sixel::bitmerge::{
    BitMergeSixelEncoder128,
    BitMergeSixelEncoder16,
    BitMergeSixelEncoder256,
    BitMergeSixelEncoder32,
    BitMergeSixelEncoder4,
    BitMergeSixelEncoder64,
    BitMergeSixelEncoder8,
    BitMergeSixelEncoderMono,
};
#[cfg(feature = "focal")]
use a_sixel::focal::{
    FocalSixelEncoder128,
    FocalSixelEncoder16,
    FocalSixelEncoder256,
    FocalSixelEncoder32,
    FocalSixelEncoder4,
    FocalSixelEncoder64,
    FocalSixelEncoder8,
    FocalSixelEncoderMono,
};
#[cfg(feature = "k-means")]
use a_sixel::kmeans::{
    KMeansSixelEncoder128,
    KMeansSixelEncoder16,
    KMeansSixelEncoder256,
    KMeansSixelEncoder32,
    KMeansSixelEncoder4,
    KMeansSixelEncoder64,
    KMeansSixelEncoder8,
    KMeansSixelEncoderMono,
};
#[cfg(feature = "k-medians")]
use a_sixel::kmedians::{
    KMediansSixelEncoder128,
    KMediansSixelEncoder16,
    KMediansSixelEncoder256,
    KMediansSixelEncoder32,
    KMediansSixelEncoder4,
    KMediansSixelEncoder64,
    KMediansSixelEncoder8,
    KMediansSixelEncoderMono,
};
#[cfg(feature = "median-cut")]
use a_sixel::median_cut::{
    MedianCutSixelEncoder128,
    MedianCutSixelEncoder16,
    MedianCutSixelEncoder256,
    MedianCutSixelEncoder32,
    MedianCutSixelEncoder4,
    MedianCutSixelEncoder64,
    MedianCutSixelEncoder8,
    MedianCutSixelEncoderMono,
};
#[cfg(feature = "octree")]
use a_sixel::octree::{
    OctreeSixelEncoder128,
    OctreeSixelEncoder16,
    OctreeSixelEncoder256,
    OctreeSixelEncoder32,
    OctreeSixelEncoder4,
    OctreeSixelEncoder64,
    OctreeSixelEncoder8,
    OctreeSixelEncoderMono,
};
#[cfg(feature = "wu")]
use a_sixel::wu::{
    WuSixelEncoder128,
    WuSixelEncoder16,
    WuSixelEncoder256,
    WuSixelEncoder32,
    WuSixelEncoder4,
    WuSixelEncoder64,
    WuSixelEncoder8,
    WuSixelEncoderMono,
};
use a_sixel::{
    bit::{
        BitSixelEncoder128,
        BitSixelEncoder16,
        BitSixelEncoder256,
        BitSixelEncoder32,
        BitSixelEncoder4,
        BitSixelEncoder64,
        BitSixelEncoder8,
        BitSixelEncoderMono,
    },
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
    BitMerge,
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
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let image_path = args.image_path;

    println!(
        "Loading image: {}",
        std::path::Path::new(&image_path)
            .file_stem()
            .unwrap()
            .to_string_lossy()
    );
    println!("algorithm: {}", args.palette_format);

    let timer = std::time::Instant::now();

    let image = image::load_from_memory(&read(image_path)?)?;
    let image = image.to_rgba8();

    let six = match args.palette_format {
        #[cfg(feature = "adu")]
        PaletteFormat::Adu => match args.palette_size {
            0..3 => match args.dither {
                Dither::No => <ADUSixelEncoderMono<NoDither>>::encode(image),
                Dither::Sobol => <ADUSixelEncoderMono<Sobol>>::encode(image),
                Dither::Sierra => <ADUSixelEncoderMono>::encode(image),
                Dither::Bayer => <ADUSixelEncoderMono<Bayer>>::encode(image),
            },
            3..6 => match args.dither {
                Dither::No => <ADUSixelEncoder4<NoDither>>::encode(image),
                Dither::Sobol => <ADUSixelEncoder4<Sobol>>::encode(image),
                Dither::Sierra => <ADUSixelEncoder4>::encode(image),
                Dither::Bayer => <ADUSixelEncoder4<Bayer>>::encode(image),
            },
            6..12 => match args.dither {
                Dither::No => <ADUSixelEncoder8<NoDither>>::encode(image),
                Dither::Sobol => <ADUSixelEncoder8<Sobol>>::encode(image),
                Dither::Sierra => <ADUSixelEncoder8>::encode(image),
                Dither::Bayer => <ADUSixelEncoder8<Bayer>>::encode(image),
            },
            12..24 => match args.dither {
                Dither::No => <ADUSixelEncoder16<NoDither>>::encode(image),
                Dither::Sobol => <ADUSixelEncoder16<Sobol>>::encode(image),
                Dither::Sierra => <ADUSixelEncoder16>::encode(image),
                Dither::Bayer => <ADUSixelEncoder16<Bayer>>::encode(image),
            },
            24..48 => match args.dither {
                Dither::No => <ADUSixelEncoder32<NoDither>>::encode(image),
                Dither::Sobol => <ADUSixelEncoder32<Sobol>>::encode(image),
                Dither::Sierra => <ADUSixelEncoder32>::encode(image),
                Dither::Bayer => <ADUSixelEncoder32<Bayer>>::encode(image),
            },
            48..86 => match args.dither {
                Dither::No => <ADUSixelEncoder64<NoDither>>::encode(image),
                Dither::Sobol => <ADUSixelEncoder64<Sobol>>::encode(image),
                Dither::Sierra => <ADUSixelEncoder64>::encode(image),
                Dither::Bayer => <ADUSixelEncoder64<Bayer>>::encode(image),
            },
            86..192 => match args.dither {
                Dither::No => <ADUSixelEncoder128<NoDither>>::encode(image),
                Dither::Sobol => <ADUSixelEncoder128<Sobol>>::encode(image),
                Dither::Sierra => <ADUSixelEncoder128>::encode(image),
                Dither::Bayer => <ADUSixelEncoder128<Bayer>>::encode(image),
            },
            _ => match args.dither {
                Dither::No => <ADUSixelEncoder256<NoDither>>::encode(image),
                Dither::Sobol => <ADUSixelEncoder256<Sobol>>::encode(image),
                Dither::Sierra => <ADUSixelEncoder256>::encode(image),
                Dither::Bayer => <ADUSixelEncoder256<Bayer>>::encode(image),
            },
        },
        PaletteFormat::Bit => match args.palette_size {
            0..3 => match args.dither {
                Dither::No => <BitSixelEncoderMono<NoDither>>::encode(image),
                Dither::Sobol => <BitSixelEncoderMono<Sobol>>::encode(image),
                Dither::Sierra => <BitSixelEncoderMono>::encode(image),
                Dither::Bayer => <BitSixelEncoderMono<Bayer>>::encode(image),
            },
            3..6 => match args.dither {
                Dither::No => <BitSixelEncoder4<NoDither>>::encode(image),
                Dither::Sobol => <BitSixelEncoder4<Sobol>>::encode(image),
                Dither::Sierra => <BitSixelEncoder4>::encode(image),
                Dither::Bayer => <BitSixelEncoder4<Bayer>>::encode(image),
            },
            6..12 => match args.dither {
                Dither::No => <BitSixelEncoder8<NoDither>>::encode(image),
                Dither::Sobol => <BitSixelEncoder8<Sobol>>::encode(image),
                Dither::Sierra => <BitSixelEncoder8>::encode(image),
                Dither::Bayer => <BitSixelEncoder8<Bayer>>::encode(image),
            },
            12..24 => match args.dither {
                Dither::No => <BitSixelEncoder16<NoDither>>::encode(image),
                Dither::Sobol => <BitSixelEncoder16<Sobol>>::encode(image),
                Dither::Sierra => <BitSixelEncoder16>::encode(image),
                Dither::Bayer => <BitSixelEncoder16<Bayer>>::encode(image),
            },
            24..48 => match args.dither {
                Dither::No => <BitSixelEncoder32<NoDither>>::encode(image),
                Dither::Sobol => <BitSixelEncoder32<Sobol>>::encode(image),
                Dither::Sierra => <BitSixelEncoder32>::encode(image),
                Dither::Bayer => <BitSixelEncoder32<Bayer>>::encode(image),
            },
            48..86 => match args.dither {
                Dither::No => <BitSixelEncoder64<NoDither>>::encode(image),
                Dither::Sobol => <BitSixelEncoder64<Sobol>>::encode(image),
                Dither::Sierra => <BitSixelEncoder64>::encode(image),
                Dither::Bayer => <BitSixelEncoder64<Bayer>>::encode(image),
            },
            86..192 => match args.dither {
                Dither::No => <BitSixelEncoder128<NoDither>>::encode(image),
                Dither::Sobol => <BitSixelEncoder128<Sobol>>::encode(image),
                Dither::Sierra => <BitSixelEncoder128>::encode(image),
                Dither::Bayer => <BitSixelEncoder128<Bayer>>::encode(image),
            },
            _ => match args.dither {
                Dither::No => <BitSixelEncoder256<NoDither>>::encode(image),
                Dither::Sobol => <BitSixelEncoder256<Sobol>>::encode(image),
                Dither::Sierra => <BitSixelEncoder256>::encode(image),
                Dither::Bayer => <BitSixelEncoder256<Bayer>>::encode(image),
            },
        },
        #[cfg(feature = "bit-merge")]
        PaletteFormat::BitMerge => match args.palette_size {
            0..3 => match args.dither {
                Dither::No => <BitMergeSixelEncoderMono<NoDither>>::encode(image),
                Dither::Sobol => <BitMergeSixelEncoderMono<Sobol>>::encode(image),
                Dither::Sierra => <BitMergeSixelEncoderMono>::encode(image),
                Dither::Bayer => <BitMergeSixelEncoderMono<Bayer>>::encode(image),
            },
            3..6 => match args.dither {
                Dither::No => <BitMergeSixelEncoder4<NoDither>>::encode(image),
                Dither::Sobol => <BitMergeSixelEncoder4<Sobol>>::encode(image),
                Dither::Sierra => <BitMergeSixelEncoder4>::encode(image),
                Dither::Bayer => <BitMergeSixelEncoder4<Bayer>>::encode(image),
            },
            6..12 => match args.dither {
                Dither::No => <BitMergeSixelEncoder8<NoDither>>::encode(image),
                Dither::Sobol => <BitMergeSixelEncoder8<Sobol>>::encode(image),
                Dither::Sierra => <BitMergeSixelEncoder8>::encode(image),
                Dither::Bayer => <BitMergeSixelEncoder8<Bayer>>::encode(image),
            },
            12..24 => match args.dither {
                Dither::No => <BitMergeSixelEncoder16<NoDither>>::encode(image),
                Dither::Sobol => <BitMergeSixelEncoder16<Sobol>>::encode(image),
                Dither::Sierra => <BitMergeSixelEncoder16>::encode(image),
                Dither::Bayer => <BitMergeSixelEncoder16<Bayer>>::encode(image),
            },
            24..48 => match args.dither {
                Dither::No => <BitMergeSixelEncoder32<NoDither>>::encode(image),
                Dither::Sobol => <BitMergeSixelEncoder32<Sobol>>::encode(image),
                Dither::Sierra => <BitMergeSixelEncoder32>::encode(image),
                Dither::Bayer => <BitMergeSixelEncoder32<Bayer>>::encode(image),
            },
            48..86 => match args.dither {
                Dither::No => <BitMergeSixelEncoder64<NoDither>>::encode(image),
                Dither::Sobol => <BitMergeSixelEncoder64<Sobol>>::encode(image),
                Dither::Sierra => <BitMergeSixelEncoder64>::encode(image),
                Dither::Bayer => <BitMergeSixelEncoder64<Bayer>>::encode(image),
            },
            86..192 => match args.dither {
                Dither::No => <BitMergeSixelEncoder128<NoDither>>::encode(image),
                Dither::Sobol => <BitMergeSixelEncoder128<Sobol>>::encode(image),
                Dither::Sierra => <BitMergeSixelEncoder128>::encode(image),
                Dither::Bayer => <BitMergeSixelEncoder128<Bayer>>::encode(image),
            },
            _ => match args.dither {
                Dither::No => <BitMergeSixelEncoder256<NoDither>>::encode(image),
                Dither::Sobol => <BitMergeSixelEncoder256<Sobol>>::encode(image),
                Dither::Sierra => <BitMergeSixelEncoder256>::encode(image),
                Dither::Bayer => <BitMergeSixelEncoder256<Bayer>>::encode(image),
            },
        },
        #[cfg(feature = "focal")]
        PaletteFormat::Focal => match args.palette_size {
            0..3 => match args.dither {
                Dither::No => <FocalSixelEncoderMono<NoDither>>::encode(image),
                Dither::Sobol => <FocalSixelEncoderMono<Sobol>>::encode(image),
                Dither::Sierra => <FocalSixelEncoderMono>::encode(image),
                Dither::Bayer => <FocalSixelEncoderMono<Bayer>>::encode(image),
            },
            3..6 => match args.dither {
                Dither::No => <FocalSixelEncoder4<NoDither>>::encode(image),
                Dither::Sobol => <FocalSixelEncoder4<Sobol>>::encode(image),
                Dither::Sierra => <FocalSixelEncoder4>::encode(image),
                Dither::Bayer => <FocalSixelEncoder4<Bayer>>::encode(image),
            },
            6..12 => match args.dither {
                Dither::No => <FocalSixelEncoder8<NoDither>>::encode(image),
                Dither::Sobol => <FocalSixelEncoder8<Sobol>>::encode(image),
                Dither::Sierra => <FocalSixelEncoder8>::encode(image),
                Dither::Bayer => <FocalSixelEncoder8<Bayer>>::encode(image),
            },
            12..24 => match args.dither {
                Dither::No => <FocalSixelEncoder16<NoDither>>::encode(image),
                Dither::Sobol => <FocalSixelEncoder16<Sobol>>::encode(image),
                Dither::Sierra => <FocalSixelEncoder16>::encode(image),
                Dither::Bayer => <FocalSixelEncoder16<Bayer>>::encode(image),
            },
            24..48 => match args.dither {
                Dither::No => <FocalSixelEncoder32<NoDither>>::encode(image),
                Dither::Sobol => <FocalSixelEncoder32<Sobol>>::encode(image),
                Dither::Sierra => <FocalSixelEncoder32>::encode(image),
                Dither::Bayer => <FocalSixelEncoder32<Bayer>>::encode(image),
            },
            48..86 => match args.dither {
                Dither::No => <FocalSixelEncoder64<NoDither>>::encode(image),
                Dither::Sobol => <FocalSixelEncoder64<Sobol>>::encode(image),
                Dither::Sierra => <FocalSixelEncoder64>::encode(image),
                Dither::Bayer => <FocalSixelEncoder64<Bayer>>::encode(image),
            },
            86..192 => match args.dither {
                Dither::No => <FocalSixelEncoder128<NoDither>>::encode(image),
                Dither::Sobol => <FocalSixelEncoder128<Sobol>>::encode(image),
                Dither::Sierra => <FocalSixelEncoder128>::encode(image),
                Dither::Bayer => <FocalSixelEncoder128<Bayer>>::encode(image),
            },
            _ => match args.dither {
                Dither::No => <FocalSixelEncoder256<NoDither>>::encode(image),
                Dither::Sobol => <FocalSixelEncoder256<Sobol>>::encode(image),
                Dither::Sierra => <FocalSixelEncoder256>::encode(image),
                Dither::Bayer => <FocalSixelEncoder256<Bayer>>::encode(image),
            },
        },
        #[cfg(feature = "k-means")]
        PaletteFormat::KMeans => match args.palette_size {
            0..3 => match args.dither {
                Dither::No => <KMeansSixelEncoderMono<NoDither>>::encode(image),
                Dither::Sobol => <KMeansSixelEncoderMono<Sobol>>::encode(image),
                Dither::Sierra => <KMeansSixelEncoderMono>::encode(image),
                Dither::Bayer => <KMeansSixelEncoderMono<Bayer>>::encode(image),
            },
            3..6 => match args.dither {
                Dither::No => <KMeansSixelEncoder4<NoDither>>::encode(image),
                Dither::Sobol => <KMeansSixelEncoder4<Sobol>>::encode(image),
                Dither::Sierra => <KMeansSixelEncoder4>::encode(image),
                Dither::Bayer => <KMeansSixelEncoder4<Bayer>>::encode(image),
            },
            6..12 => match args.dither {
                Dither::No => <KMeansSixelEncoder8<NoDither>>::encode(image),
                Dither::Sobol => <KMeansSixelEncoder8<Sobol>>::encode(image),
                Dither::Sierra => <KMeansSixelEncoder8>::encode(image),
                Dither::Bayer => <KMeansSixelEncoder8<Bayer>>::encode(image),
            },
            12..24 => match args.dither {
                Dither::No => <KMeansSixelEncoder16<NoDither>>::encode(image),
                Dither::Sobol => <KMeansSixelEncoder16<Sobol>>::encode(image),
                Dither::Sierra => <KMeansSixelEncoder16>::encode(image),
                Dither::Bayer => <KMeansSixelEncoder16<Bayer>>::encode(image),
            },
            24..48 => match args.dither {
                Dither::No => <KMeansSixelEncoder32<NoDither>>::encode(image),
                Dither::Sobol => <KMeansSixelEncoder32<Sobol>>::encode(image),
                Dither::Sierra => <KMeansSixelEncoder32>::encode(image),
                Dither::Bayer => <KMeansSixelEncoder32<Bayer>>::encode(image),
            },
            48..86 => match args.dither {
                Dither::No => <KMeansSixelEncoder64<NoDither>>::encode(image),
                Dither::Sobol => <KMeansSixelEncoder64<Sobol>>::encode(image),
                Dither::Sierra => <KMeansSixelEncoder64>::encode(image),
                Dither::Bayer => <KMeansSixelEncoder64<Bayer>>::encode(image),
            },
            86..192 => match args.dither {
                Dither::No => <KMeansSixelEncoder128<NoDither>>::encode(image),
                Dither::Sobol => <KMeansSixelEncoder128<Sobol>>::encode(image),
                Dither::Sierra => <KMeansSixelEncoder128>::encode(image),
                Dither::Bayer => <KMeansSixelEncoder128<Bayer>>::encode(image),
            },
            _ => match args.dither {
                Dither::No => <KMeansSixelEncoder256<NoDither>>::encode(image),
                Dither::Sobol => <KMeansSixelEncoder256<Sobol>>::encode(image),
                Dither::Sierra => <KMeansSixelEncoder256>::encode(image),
                Dither::Bayer => <KMeansSixelEncoder256<Bayer>>::encode(image),
            },
        },
        #[cfg(feature = "k-medians")]
        PaletteFormat::KMedians => match args.palette_size {
            0..3 => match args.dither {
                Dither::No => <KMediansSixelEncoderMono<NoDither>>::encode(image),
                Dither::Sobol => <KMediansSixelEncoderMono<Sobol>>::encode(image),
                Dither::Sierra => <KMediansSixelEncoderMono>::encode(image),
                Dither::Bayer => <KMediansSixelEncoderMono<Bayer>>::encode(image),
            },
            3..6 => match args.dither {
                Dither::No => <KMediansSixelEncoder4<NoDither>>::encode(image),
                Dither::Sobol => <KMediansSixelEncoder4<Sobol>>::encode(image),
                Dither::Sierra => <KMediansSixelEncoder4>::encode(image),
                Dither::Bayer => <KMediansSixelEncoder4<Bayer>>::encode(image),
            },
            6..12 => match args.dither {
                Dither::No => <KMediansSixelEncoder8<NoDither>>::encode(image),
                Dither::Sobol => <KMediansSixelEncoder8<Sobol>>::encode(image),
                Dither::Sierra => <KMediansSixelEncoder8>::encode(image),
                Dither::Bayer => <KMediansSixelEncoder8<Bayer>>::encode(image),
            },
            12..24 => match args.dither {
                Dither::No => <KMediansSixelEncoder16<NoDither>>::encode(image),
                Dither::Sobol => <KMediansSixelEncoder16<Sobol>>::encode(image),
                Dither::Sierra => <KMediansSixelEncoder16>::encode(image),
                Dither::Bayer => <KMediansSixelEncoder16<Bayer>>::encode(image),
            },
            24..48 => match args.dither {
                Dither::No => <KMediansSixelEncoder32<NoDither>>::encode(image),
                Dither::Sobol => <KMediansSixelEncoder32<Sobol>>::encode(image),
                Dither::Sierra => <KMediansSixelEncoder32>::encode(image),
                Dither::Bayer => <KMediansSixelEncoder32<Bayer>>::encode(image),
            },
            48..86 => match args.dither {
                Dither::No => <KMediansSixelEncoder64<NoDither>>::encode(image),
                Dither::Sobol => <KMediansSixelEncoder64<Sobol>>::encode(image),
                Dither::Sierra => <KMediansSixelEncoder64>::encode(image),
                Dither::Bayer => <KMediansSixelEncoder64<Bayer>>::encode(image),
            },
            86..192 => match args.dither {
                Dither::No => <KMediansSixelEncoder128<NoDither>>::encode(image),
                Dither::Sobol => <KMediansSixelEncoder128<Sobol>>::encode(image),
                Dither::Sierra => <KMediansSixelEncoder128>::encode(image),
                Dither::Bayer => <KMediansSixelEncoder128<Bayer>>::encode(image),
            },
            _ => match args.dither {
                Dither::No => <KMediansSixelEncoder256<NoDither>>::encode(image),
                Dither::Sobol => <KMediansSixelEncoder256<Sobol>>::encode(image),
                Dither::Sierra => <KMediansSixelEncoder256>::encode(image),
                Dither::Bayer => <KMediansSixelEncoder256<Bayer>>::encode(image),
            },
        },
        #[cfg(feature = "median-cut")]
        PaletteFormat::MedianCut => match args.palette_size {
            0..3 => match args.dither {
                Dither::No => <MedianCutSixelEncoderMono<NoDither>>::encode(image),
                Dither::Sobol => <MedianCutSixelEncoderMono<Sobol>>::encode(image),
                Dither::Sierra => <MedianCutSixelEncoderMono>::encode(image),
                Dither::Bayer => <MedianCutSixelEncoderMono<Bayer>>::encode(image),
            },
            3..6 => match args.dither {
                Dither::No => <MedianCutSixelEncoder4<NoDither>>::encode(image),
                Dither::Sobol => <MedianCutSixelEncoder4<Sobol>>::encode(image),
                Dither::Sierra => <MedianCutSixelEncoder4>::encode(image),
                Dither::Bayer => <MedianCutSixelEncoder4<Bayer>>::encode(image),
            },
            6..12 => match args.dither {
                Dither::No => <MedianCutSixelEncoder8<NoDither>>::encode(image),
                Dither::Sobol => <MedianCutSixelEncoder8<Sobol>>::encode(image),
                Dither::Sierra => <MedianCutSixelEncoder8>::encode(image),
                Dither::Bayer => <MedianCutSixelEncoder8<Bayer>>::encode(image),
            },
            12..24 => match args.dither {
                Dither::No => <MedianCutSixelEncoder16<NoDither>>::encode(image),
                Dither::Sobol => <MedianCutSixelEncoder16<Sobol>>::encode(image),
                Dither::Sierra => <MedianCutSixelEncoder16>::encode(image),
                Dither::Bayer => <MedianCutSixelEncoder16<Bayer>>::encode(image),
            },
            24..48 => match args.dither {
                Dither::No => <MedianCutSixelEncoder32<NoDither>>::encode(image),
                Dither::Sobol => <MedianCutSixelEncoder32<Sobol>>::encode(image),
                Dither::Sierra => <MedianCutSixelEncoder32>::encode(image),
                Dither::Bayer => <MedianCutSixelEncoder32<Bayer>>::encode(image),
            },
            48..86 => match args.dither {
                Dither::No => <MedianCutSixelEncoder64<NoDither>>::encode(image),
                Dither::Sobol => <MedianCutSixelEncoder64<Sobol>>::encode(image),
                Dither::Sierra => <MedianCutSixelEncoder64>::encode(image),
                Dither::Bayer => <MedianCutSixelEncoder64<Bayer>>::encode(image),
            },
            86..192 => match args.dither {
                Dither::No => <MedianCutSixelEncoder128<NoDither>>::encode(image),
                Dither::Sobol => <MedianCutSixelEncoder128<Sobol>>::encode(image),
                Dither::Sierra => <MedianCutSixelEncoder128>::encode(image),
                Dither::Bayer => <MedianCutSixelEncoder128<Bayer>>::encode(image),
            },
            _ => match args.dither {
                Dither::No => <MedianCutSixelEncoder256<NoDither>>::encode(image),
                Dither::Sobol => <MedianCutSixelEncoder256<Sobol>>::encode(image),
                Dither::Sierra => <MedianCutSixelEncoder256>::encode(image),
                Dither::Bayer => <MedianCutSixelEncoder256<Bayer>>::encode(image),
            },
        },
        #[cfg(feature = "octree")]
        PaletteFormat::Octree => match args.palette_size {
            0..3 => match args.dither {
                Dither::No => <OctreeSixelEncoderMono<NoDither>>::encode(image),
                Dither::Sobol => <OctreeSixelEncoderMono<Sobol>>::encode(image),
                Dither::Sierra => <OctreeSixelEncoderMono>::encode(image),
                Dither::Bayer => <OctreeSixelEncoderMono<Bayer>>::encode(image),
            },
            3..6 => match args.dither {
                Dither::No => <OctreeSixelEncoder4<NoDither>>::encode(image),
                Dither::Sobol => <OctreeSixelEncoder4<Sobol>>::encode(image),
                Dither::Sierra => <OctreeSixelEncoder4>::encode(image),
                Dither::Bayer => <OctreeSixelEncoder4<Bayer>>::encode(image),
            },
            6..12 => match args.dither {
                Dither::No => <OctreeSixelEncoder8<NoDither>>::encode(image),
                Dither::Sobol => <OctreeSixelEncoder8<Sobol>>::encode(image),
                Dither::Sierra => <OctreeSixelEncoder8>::encode(image),
                Dither::Bayer => <OctreeSixelEncoder8<Bayer>>::encode(image),
            },
            12..24 => match args.dither {
                Dither::No => <OctreeSixelEncoder16<NoDither>>::encode(image),
                Dither::Sobol => <OctreeSixelEncoder16<Sobol>>::encode(image),
                Dither::Sierra => <OctreeSixelEncoder16>::encode(image),
                Dither::Bayer => <OctreeSixelEncoder16<Bayer>>::encode(image),
            },
            24..48 => match args.dither {
                Dither::No => <OctreeSixelEncoder32<NoDither>>::encode(image),
                Dither::Sobol => <OctreeSixelEncoder32<Sobol>>::encode(image),
                Dither::Sierra => <OctreeSixelEncoder32>::encode(image),
                Dither::Bayer => <OctreeSixelEncoder32<Bayer>>::encode(image),
            },
            48..86 => match args.dither {
                Dither::No => <OctreeSixelEncoder64<NoDither>>::encode(image),
                Dither::Sobol => <OctreeSixelEncoder64<Sobol>>::encode(image),
                Dither::Sierra => <OctreeSixelEncoder64>::encode(image),
                Dither::Bayer => <OctreeSixelEncoder64<Bayer>>::encode(image),
            },
            86..192 => match args.dither {
                Dither::No => <OctreeSixelEncoder128<NoDither>>::encode(image),
                Dither::Sobol => <OctreeSixelEncoder128<Sobol>>::encode(image),
                Dither::Sierra => <OctreeSixelEncoder128>::encode(image),
                Dither::Bayer => <OctreeSixelEncoder128<Bayer>>::encode(image),
            },
            _ => match args.dither {
                Dither::No => <OctreeSixelEncoder256<NoDither>>::encode(image),
                Dither::Sobol => <OctreeSixelEncoder256<Sobol>>::encode(image),
                Dither::Sierra => <OctreeSixelEncoder256>::encode(image),
                Dither::Bayer => <OctreeSixelEncoder256<Bayer>>::encode(image),
            },
        },
        #[cfg(feature = "wu")]
        PaletteFormat::Wu => match args.palette_size {
            0..3 => match args.dither {
                Dither::No => <WuSixelEncoderMono<NoDither>>::encode(image),
                Dither::Sobol => <WuSixelEncoderMono<Sobol>>::encode(image),
                Dither::Sierra => <WuSixelEncoderMono>::encode(image),
                Dither::Bayer => <WuSixelEncoderMono<Bayer>>::encode(image),
            },
            3..6 => match args.dither {
                Dither::No => <WuSixelEncoder4<NoDither>>::encode(image),
                Dither::Sobol => <WuSixelEncoder4<Sobol>>::encode(image),
                Dither::Sierra => <WuSixelEncoder4>::encode(image),
                Dither::Bayer => <WuSixelEncoder4<Bayer>>::encode(image),
            },
            6..12 => match args.dither {
                Dither::No => <WuSixelEncoder8<NoDither>>::encode(image),
                Dither::Sobol => <WuSixelEncoder8<Sobol>>::encode(image),
                Dither::Sierra => <WuSixelEncoder8>::encode(image),
                Dither::Bayer => <WuSixelEncoder8<Bayer>>::encode(image),
            },
            12..24 => match args.dither {
                Dither::No => <WuSixelEncoder16<NoDither>>::encode(image),
                Dither::Sobol => <WuSixelEncoder16<Sobol>>::encode(image),
                Dither::Sierra => <WuSixelEncoder16>::encode(image),
                Dither::Bayer => <WuSixelEncoder16<Bayer>>::encode(image),
            },
            24..48 => match args.dither {
                Dither::No => <WuSixelEncoder32<NoDither>>::encode(image),
                Dither::Sobol => <WuSixelEncoder32<Sobol>>::encode(image),
                Dither::Sierra => <WuSixelEncoder32>::encode(image),
                Dither::Bayer => <WuSixelEncoder32<Bayer>>::encode(image),
            },
            48..86 => match args.dither {
                Dither::No => <WuSixelEncoder64<NoDither>>::encode(image),
                Dither::Sobol => <WuSixelEncoder64<Sobol>>::encode(image),
                Dither::Sierra => <WuSixelEncoder64>::encode(image),
                Dither::Bayer => <WuSixelEncoder64<Bayer>>::encode(image),
            },
            86..192 => match args.dither {
                Dither::No => <WuSixelEncoder128<NoDither>>::encode(image),
                Dither::Sobol => <WuSixelEncoder128<Sobol>>::encode(image),
                Dither::Sierra => <WuSixelEncoder128>::encode(image),
                Dither::Bayer => <WuSixelEncoder128<Bayer>>::encode(image),
            },
            _ => match args.dither {
                Dither::No => <WuSixelEncoder256<NoDither>>::encode(image),
                Dither::Sobol => <WuSixelEncoder256<Sobol>>::encode(image),
                Dither::Sierra => <WuSixelEncoder256>::encode(image),
                Dither::Bayer => <WuSixelEncoder256<Bayer>>::encode(image),
            },
        },
    };

    println!("Time taken: {}ms", timer.elapsed().as_millis());

    println!("{six}");

    Ok(())
}
