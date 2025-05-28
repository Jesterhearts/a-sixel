use std::fs::read;

use a_sixel::{
    adu::{
        ADUSixelEncoder128,
        ADUSixelEncoder16,
        ADUSixelEncoder256,
        ADUSixelEncoder32,
        ADUSixelEncoder4,
        ADUSixelEncoder64,
        ADUSixelEncoder8,
        ADUSixelEncoderMono,
    },
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
        NoDither,
        Sierra,
        Sobol,
    },
    focal::{
        FocalSixelEncoder128,
        FocalSixelEncoder16,
        FocalSixelEncoder256,
        FocalSixelEncoder32,
        FocalSixelEncoder4,
        FocalSixelEncoder64,
        FocalSixelEncoder8,
        FocalSixelEncoderMono,
    },
    kmeans::{
        KMeansSixelEncoder128,
        KMeansSixelEncoder16,
        KMeansSixelEncoder256,
        KMeansSixelEncoder32,
        KMeansSixelEncoder4,
        KMeansSixelEncoder64,
        KMeansSixelEncoder8,
        KMeansSixelEncoderMono,
    },
    kmedians::{
        KMediansSixelEncoder128,
        KMediansSixelEncoder16,
        KMediansSixelEncoder256,
        KMediansSixelEncoder32,
        KMediansSixelEncoder4,
        KMediansSixelEncoder64,
        KMediansSixelEncoder8,
        KMediansSixelEncoderMono,
    },
    median_cut::{
        MedianCutSixelEncoder128,
        MedianCutSixelEncoder16,
        MedianCutSixelEncoder256,
        MedianCutSixelEncoder32,
        MedianCutSixelEncoder4,
        MedianCutSixelEncoder64,
        MedianCutSixelEncoder8,
        MedianCutSixelEncoderMono,
    },
    octree::{
        OctreeSixelEncoder128,
        OctreeSixelEncoder16,
        OctreeSixelEncoder256,
        OctreeSixelEncoder32,
        OctreeSixelEncoder4,
        OctreeSixelEncoder64,
        OctreeSixelEncoder8,
        OctreeSixelEncoderMono,
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
    Adu,
    Focal,
    MedianCut,
    Bit,
    Octree,
    #[strum(serialize = "kmeans", serialize = "k-means")]
    KMeans,
    #[strum(
        serialize = "kmedians",
        serialize = "kmed",
        serialize = "k-medians",
        serialize = "k-med"
    )]
    KMedianss,
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
    #[clap(long, short = 'f', default_value_t = PaletteFormat::Focal)]
    palette_format: PaletteFormat,

    /// Whether to use Sobol dithering instead of the default dithering.
    #[clap(long, short, default_value_t = false)]
    sobol: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let image_path = args.image_path;

    let image = image::load_from_memory(&read(image_path)?)?;
    let image = image.to_rgb8();

    let six = match args.palette_format {
        PaletteFormat::Adu => match args.palette_size {
            0..3 => {
                if args.sobol {
                    <ADUSixelEncoderMono<Sobol>>::encode(&image)
                } else {
                    <ADUSixelEncoderMono>::encode(&image)
                }
            }
            3..6 => {
                if args.sobol {
                    <ADUSixelEncoder4<Sobol>>::encode(&image)
                } else {
                    <ADUSixelEncoder4>::encode(&image)
                }
            }
            6..12 => {
                if args.sobol {
                    <ADUSixelEncoder8<Sobol>>::encode(&image)
                } else {
                    <ADUSixelEncoder8>::encode(&image)
                }
            }
            12..24 => {
                if args.sobol {
                    <ADUSixelEncoder16<Sobol>>::encode(&image)
                } else {
                    <ADUSixelEncoder16>::encode(&image)
                }
            }
            24..48 => {
                if args.sobol {
                    <ADUSixelEncoder32<Sobol>>::encode(&image)
                } else {
                    <ADUSixelEncoder32>::encode(&image)
                }
            }
            48..86 => {
                if args.sobol {
                    <ADUSixelEncoder64<Sobol>>::encode(&image)
                } else {
                    <ADUSixelEncoder64>::encode(&image)
                }
            }
            86..192 => {
                if args.sobol {
                    <ADUSixelEncoder128<Sobol>>::encode(&image)
                } else {
                    <ADUSixelEncoder128>::encode(&image)
                }
            }
            _ => {
                if args.sobol {
                    <ADUSixelEncoder256<Sobol>>::encode(&image)
                } else {
                    <ADUSixelEncoder256>::encode(&image)
                }
            }
        },
        PaletteFormat::Focal => match args.palette_size {
            0..3 => {
                if args.sobol {
                    <FocalSixelEncoderMono<Sobol>>::encode(&image)
                } else {
                    <FocalSixelEncoderMono>::encode(&image)
                }
            }
            3..6 => {
                if args.sobol {
                    <FocalSixelEncoder4<Sobol>>::encode(&image)
                } else {
                    <FocalSixelEncoder4>::encode(&image)
                }
            }
            6..12 => {
                if args.sobol {
                    <FocalSixelEncoder8<Sobol>>::encode(&image)
                } else {
                    <FocalSixelEncoder8>::encode(&image)
                }
            }
            12..24 => {
                if args.sobol {
                    <FocalSixelEncoder16<Sobol>>::encode(&image)
                } else {
                    <FocalSixelEncoder16>::encode(&image)
                }
            }
            24..48 => {
                if args.sobol {
                    <FocalSixelEncoder32<Sobol>>::encode(&image)
                } else {
                    <FocalSixelEncoder32>::encode(&image)
                }
            }
            48..86 => {
                if args.sobol {
                    <FocalSixelEncoder64<Sobol>>::encode(&image)
                } else {
                    <FocalSixelEncoder64>::encode(&image)
                }
            }
            86..192 => {
                if args.sobol {
                    <FocalSixelEncoder128<Sobol>>::encode(&image)
                } else {
                    <FocalSixelEncoder128>::encode(&image)
                }
            }
            _ => {
                if args.sobol {
                    <FocalSixelEncoder256<Sobol>>::encode(&image)
                } else {
                    <FocalSixelEncoder256>::encode(&image)
                }
            }
        },
        PaletteFormat::MedianCut => match args.palette_size {
            0..3 => {
                if args.sobol {
                    <MedianCutSixelEncoderMono<Sobol>>::encode(&image)
                } else {
                    <MedianCutSixelEncoderMono>::encode(&image)
                }
            }
            3..6 => {
                if args.sobol {
                    <MedianCutSixelEncoder4<Sobol>>::encode(&image)
                } else {
                    <MedianCutSixelEncoder4>::encode(&image)
                }
            }
            6..12 => {
                if args.sobol {
                    <MedianCutSixelEncoder8<Sobol>>::encode(&image)
                } else {
                    <MedianCutSixelEncoder8>::encode(&image)
                }
            }
            12..24 => {
                if args.sobol {
                    <MedianCutSixelEncoder16<Sobol>>::encode(&image)
                } else {
                    <MedianCutSixelEncoder16>::encode(&image)
                }
            }
            24..48 => {
                if args.sobol {
                    <MedianCutSixelEncoder32<Sobol>>::encode(&image)
                } else {
                    <MedianCutSixelEncoder32>::encode(&image)
                }
            }
            48..86 => {
                if args.sobol {
                    <MedianCutSixelEncoder64<Sobol>>::encode(&image)
                } else {
                    <MedianCutSixelEncoder64>::encode(&image)
                }
            }
            86..192 => {
                if args.sobol {
                    <MedianCutSixelEncoder128<Sobol>>::encode(&image)
                } else {
                    <MedianCutSixelEncoder128>::encode(&image)
                }
            }
            _ => {
                if args.sobol {
                    <MedianCutSixelEncoder256<Sobol>>::encode(&image)
                } else {
                    <MedianCutSixelEncoder256>::encode(&image)
                }
            }
        },
        PaletteFormat::Bit => match args.palette_size {
            0 => <BitSixelEncoder256<NoDither>>::encode(&image),
            1..3 => {
                if args.sobol {
                    <BitSixelEncoderMono<Sobol>>::encode(&image)
                } else {
                    <BitSixelEncoderMono>::encode(&image)
                }
            }
            3..6 => {
                if args.sobol {
                    <BitSixelEncoder4<Sobol>>::encode(&image)
                } else {
                    <BitSixelEncoder4>::encode(&image)
                }
            }
            6..12 => {
                if args.sobol {
                    <BitSixelEncoder8<Sobol>>::encode(&image)
                } else {
                    <BitSixelEncoder8>::encode(&image)
                }
            }
            12..24 => {
                if args.sobol {
                    <BitSixelEncoder16<Sobol>>::encode(&image)
                } else {
                    <BitSixelEncoder16>::encode(&image)
                }
            }
            24..48 => {
                if args.sobol {
                    <BitSixelEncoder32<Sobol>>::encode(&image)
                } else {
                    <BitSixelEncoder32>::encode(&image)
                }
            }
            48..86 => {
                if args.sobol {
                    <BitSixelEncoder64<Sobol>>::encode(&image)
                } else {
                    <BitSixelEncoder64>::encode(&image)
                }
            }
            86..192 => {
                if args.sobol {
                    <BitSixelEncoder128<Sobol>>::encode(&image)
                } else {
                    <BitSixelEncoder128>::encode(&image)
                }
            }
            _ => {
                if args.sobol {
                    <BitSixelEncoder256<Sobol>>::encode(&image)
                } else {
                    <BitSixelEncoder256>::encode(&image)
                }
            }
        },
        PaletteFormat::Octree => match args.palette_size {
            0..3 => {
                if args.sobol {
                    <OctreeSixelEncoderMono<Sobol>>::encode(&image)
                } else {
                    <OctreeSixelEncoderMono>::encode(&image)
                }
            }
            3..6 => {
                if args.sobol {
                    <OctreeSixelEncoder4<Sobol>>::encode(&image)
                } else {
                    <OctreeSixelEncoder4>::encode(&image)
                }
            }
            6..12 => {
                if args.sobol {
                    <OctreeSixelEncoder8<Sobol>>::encode(&image)
                } else {
                    <OctreeSixelEncoder8>::encode(&image)
                }
            }
            12..24 => {
                if args.sobol {
                    <OctreeSixelEncoder16<Sobol>>::encode(&image)
                } else {
                    OctreeSixelEncoder16::<Sierra, true>::encode(&image)
                }
            }
            24..48 => {
                if args.sobol {
                    <OctreeSixelEncoder32<Sobol>>::encode(&image)
                } else {
                    <OctreeSixelEncoder32>::encode(&image)
                }
            }
            48..86 => {
                if args.sobol {
                    <OctreeSixelEncoder64<Sobol>>::encode(&image)
                } else {
                    <OctreeSixelEncoder64>::encode(&image)
                }
            }
            86..192 => {
                if args.sobol {
                    <OctreeSixelEncoder128<Sobol>>::encode(&image)
                } else {
                    <OctreeSixelEncoder128>::encode(&image)
                }
            }
            _ => {
                if args.sobol {
                    <OctreeSixelEncoder256<Sobol>>::encode(&image)
                } else {
                    <OctreeSixelEncoder256>::encode(&image)
                }
            }
        },
        PaletteFormat::KMeans => match args.palette_size {
            0..3 => {
                if args.sobol {
                    <KMeansSixelEncoderMono<Sobol>>::encode(&image)
                } else {
                    <KMeansSixelEncoderMono>::encode(&image)
                }
            }
            3..6 => {
                if args.sobol {
                    <KMeansSixelEncoder4<Sobol>>::encode(&image)
                } else {
                    <KMeansSixelEncoder4>::encode(&image)
                }
            }
            6..12 => {
                if args.sobol {
                    <KMeansSixelEncoder8<Sobol>>::encode(&image)
                } else {
                    <KMeansSixelEncoder8>::encode(&image)
                }
            }
            12..24 => {
                if args.sobol {
                    <KMeansSixelEncoder16<Sobol>>::encode(&image)
                } else {
                    <KMeansSixelEncoder16>::encode(&image)
                }
            }
            24..48 => {
                if args.sobol {
                    <KMeansSixelEncoder32<Sobol>>::encode(&image)
                } else {
                    <KMeansSixelEncoder32>::encode(&image)
                }
            }
            48..86 => {
                if args.sobol {
                    <KMeansSixelEncoder64<Sobol>>::encode(&image)
                } else {
                    <KMeansSixelEncoder64>::encode(&image)
                }
            }
            86..192 => {
                if args.sobol {
                    <KMeansSixelEncoder128<Sobol>>::encode(&image)
                } else {
                    <KMeansSixelEncoder128>::encode(&image)
                }
            }
            _ => {
                if args.sobol {
                    <KMeansSixelEncoder256<Sobol>>::encode(&image)
                } else {
                    <KMeansSixelEncoder256>::encode(&image)
                }
            }
        },
        PaletteFormat::KMedianss => match args.palette_size {
            0..3 => {
                if args.sobol {
                    <KMediansSixelEncoderMono<Sobol>>::encode(&image)
                } else {
                    <KMediansSixelEncoderMono>::encode(&image)
                }
            }
            3..6 => {
                if args.sobol {
                    <KMediansSixelEncoder4<Sobol>>::encode(&image)
                } else {
                    <KMediansSixelEncoder4>::encode(&image)
                }
            }
            6..12 => {
                if args.sobol {
                    <KMediansSixelEncoder8<Sobol>>::encode(&image)
                } else {
                    <KMediansSixelEncoder8>::encode(&image)
                }
            }
            12..24 => {
                if args.sobol {
                    <KMediansSixelEncoder16<Sobol>>::encode(&image)
                } else {
                    <KMediansSixelEncoder16>::encode(&image)
                }
            }
            24..48 => {
                if args.sobol {
                    <KMediansSixelEncoder32<Sobol>>::encode(&image)
                } else {
                    <KMediansSixelEncoder32>::encode(&image)
                }
            }
            48..86 => {
                if args.sobol {
                    <KMediansSixelEncoder64<Sobol>>::encode(&image)
                } else {
                    <KMediansSixelEncoder64>::encode(&image)
                }
            }
            86..192 => {
                if args.sobol {
                    <KMediansSixelEncoder128<Sobol>>::encode(&image)
                } else {
                    <KMediansSixelEncoder128>::encode(&image)
                }
            }
            _ => {
                if args.sobol {
                    <KMediansSixelEncoder256<Sobol>>::encode(&image)
                } else {
                    <KMediansSixelEncoder256>::encode(&image)
                }
            }
        },
    };

    println!("{six}");

    Ok(())
}
