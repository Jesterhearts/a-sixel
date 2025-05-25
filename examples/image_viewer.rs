use std::fs::read;

use a_sixel::{
    ADUSixelEncoder8,
    ADUSixelEncoder16,
    ADUSixelEncoder32,
    ADUSixelEncoder64,
    ADUSixelEncoder128,
    ADUSixelEncoder256,
    ADUSixelEncoder256High,
    BitSixelEncoder4,
    BitSixelEncoder8,
    BitSixelEncoder16,
    BitSixelEncoder32,
    BitSixelEncoder64,
    BitSixelEncoder128,
    BitSixelEncoder256,
    BitSixelEncoderMono,
    FocalSixelEncoder4,
    FocalSixelEncoder8,
    FocalSixelEncoder16,
    FocalSixelEncoder32,
    FocalSixelEncoder64,
    FocalSixelEncoder128,
    FocalSixelEncoder256,
    FocalSixelEncoder256High,
    FocalSixelEncoderMono,
    MedianCutSixelEncoder4,
    MedianCutSixelEncoder8,
    MedianCutSixelEncoder16,
    MedianCutSixelEncoder32,
    MedianCutSixelEncoder64,
    MedianCutSixelEncoder128,
    MedianCutSixelEncoder256,
    MedianCutSixelEncoderMono,
    dither::{
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
    Adu,
    Focal,
    MedianCut,
    Bit,
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
            0..12 => {
                if args.sobol {
                    <ADUSixelEncoder8<Sobol>>::encode(image)
                } else {
                    <ADUSixelEncoder8>::encode(image)
                }
            }
            12..24 => {
                if args.sobol {
                    <ADUSixelEncoder16<Sobol>>::encode(image)
                } else {
                    <ADUSixelEncoder16>::encode(image)
                }
            }
            24..48 => {
                if args.sobol {
                    <ADUSixelEncoder32<Sobol>>::encode(image)
                } else {
                    <ADUSixelEncoder32>::encode(image)
                }
            }
            48..86 => {
                if args.sobol {
                    <ADUSixelEncoder64<Sobol>>::encode(image)
                } else {
                    <ADUSixelEncoder64>::encode(image)
                }
            }
            86..192 => {
                if args.sobol {
                    <ADUSixelEncoder128<Sobol>>::encode(image)
                } else {
                    <ADUSixelEncoder128>::encode(image)
                }
            }
            192..=256 => {
                if args.sobol {
                    <ADUSixelEncoder256<Sobol>>::encode(image)
                } else {
                    <ADUSixelEncoder256>::encode(image)
                }
            }
            _ => {
                if args.sobol {
                    <ADUSixelEncoder256High<Sobol>>::encode(image)
                } else {
                    <ADUSixelEncoder256High>::encode(image)
                }
            }
        },
        PaletteFormat::Focal => match args.palette_size {
            0..3 => {
                if args.sobol {
                    <FocalSixelEncoderMono<Sobol>>::encode(image)
                } else {
                    <FocalSixelEncoderMono>::encode(image)
                }
            }
            3..6 => {
                if args.sobol {
                    <FocalSixelEncoder4<Sobol>>::encode(image)
                } else {
                    <FocalSixelEncoder4>::encode(image)
                }
            }
            6..12 => {
                if args.sobol {
                    <FocalSixelEncoder8<Sobol>>::encode(image)
                } else {
                    <FocalSixelEncoder8>::encode(image)
                }
            }
            12..24 => {
                if args.sobol {
                    <FocalSixelEncoder16<Sobol>>::encode(image)
                } else {
                    <FocalSixelEncoder16>::encode(image)
                }
            }
            24..48 => {
                if args.sobol {
                    <FocalSixelEncoder32<Sobol>>::encode(image)
                } else {
                    <FocalSixelEncoder32>::encode(image)
                }
            }
            48..86 => {
                if args.sobol {
                    <FocalSixelEncoder64<Sobol>>::encode(image)
                } else {
                    <FocalSixelEncoder64>::encode(image)
                }
            }
            86..192 => {
                if args.sobol {
                    <FocalSixelEncoder128<Sobol>>::encode(image)
                } else {
                    <FocalSixelEncoder128>::encode(image)
                }
            }
            192..=256 => {
                if args.sobol {
                    <FocalSixelEncoder256<Sobol>>::encode(image)
                } else {
                    <FocalSixelEncoder256>::encode(image)
                }
            }
            _ => {
                if args.sobol {
                    <FocalSixelEncoder256High<Sobol>>::encode(image)
                } else {
                    <FocalSixelEncoder256High>::encode(image)
                }
            }
        },
        PaletteFormat::MedianCut => match args.palette_size {
            0..3 => {
                if args.sobol {
                    <MedianCutSixelEncoderMono<Sobol>>::encode(image)
                } else {
                    <MedianCutSixelEncoderMono>::encode(image)
                }
            }
            3..6 => {
                if args.sobol {
                    <MedianCutSixelEncoder4<Sobol>>::encode(image)
                } else {
                    <MedianCutSixelEncoder4>::encode(image)
                }
            }
            6..12 => {
                if args.sobol {
                    <MedianCutSixelEncoder8<Sobol>>::encode(image)
                } else {
                    <MedianCutSixelEncoder8>::encode(image)
                }
            }
            12..24 => {
                if args.sobol {
                    <MedianCutSixelEncoder16<Sobol>>::encode(image)
                } else {
                    <MedianCutSixelEncoder16>::encode(image)
                }
            }
            24..48 => {
                if args.sobol {
                    <MedianCutSixelEncoder32<Sobol>>::encode(image)
                } else {
                    <MedianCutSixelEncoder32>::encode(image)
                }
            }
            48..86 => {
                if args.sobol {
                    <MedianCutSixelEncoder64<Sobol>>::encode(image)
                } else {
                    <MedianCutSixelEncoder64>::encode(image)
                }
            }
            86..192 => {
                if args.sobol {
                    <MedianCutSixelEncoder128<Sobol>>::encode(image)
                } else {
                    <MedianCutSixelEncoder128>::encode(image)
                }
            }
            _ => {
                if args.sobol {
                    <MedianCutSixelEncoder256<Sobol>>::encode(image)
                } else {
                    <MedianCutSixelEncoder256>::encode(image)
                }
            }
        },
        PaletteFormat::Bit => match args.palette_size {
            0 => <BitSixelEncoder256<NoDither>>::encode(image),
            1..3 => {
                if args.sobol {
                    <BitSixelEncoderMono<Sobol>>::encode(image)
                } else {
                    <BitSixelEncoderMono>::encode(image)
                }
            }
            3..6 => {
                if args.sobol {
                    <BitSixelEncoder4<Sobol>>::encode(image)
                } else {
                    <BitSixelEncoder4>::encode(image)
                }
            }
            6..12 => {
                if args.sobol {
                    <BitSixelEncoder8<Sobol>>::encode(image)
                } else {
                    <BitSixelEncoder8>::encode(image)
                }
            }
            12..24 => {
                if args.sobol {
                    <BitSixelEncoder16<Sobol>>::encode(image)
                } else {
                    <BitSixelEncoder16>::encode(image)
                }
            }
            24..48 => {
                if args.sobol {
                    <BitSixelEncoder32<Sobol>>::encode(image)
                } else {
                    <BitSixelEncoder32>::encode(image)
                }
            }
            48..86 => {
                if args.sobol {
                    <BitSixelEncoder64<Sobol>>::encode(image)
                } else {
                    <BitSixelEncoder64>::encode(image)
                }
            }
            86..192 => {
                if args.sobol {
                    <BitSixelEncoder128<Sobol>>::encode(image)
                } else {
                    <BitSixelEncoder128>::encode(image)
                }
            }
            _ => {
                if args.sobol {
                    <BitSixelEncoder256<Sobol>>::encode(image)
                } else {
                    <BitSixelEncoder256>::encode(image)
                }
            }
        },
    };

    println!("{six}");

    Ok(())
}
