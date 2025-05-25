use std::fs::read;

use a_sixel::{
    ADUSixelEncoder8,
    ADUSixelEncoder16,
    ADUSixelEncoder32,
    ADUSixelEncoder64,
    ADUSixelEncoder128,
    ADUSixelEncoder256,
    FocalSixelEncoder4,
    FocalSixelEncoder8,
    FocalSixelEncoder16,
    FocalSixelEncoder32,
    FocalSixelEncoder64,
    FocalSixelEncoder128,
    FocalSixelEncoder256,
    FocalSixelEncoderMono,
    MedianCutSixelEncoder4,
    MedianCutSixelEncoder8,
    MedianCutSixelEncoder16,
    MedianCutSixelEncoder32,
    MedianCutSixelEncoder64,
    MedianCutSixelEncoder128,
    MedianCutSixelEncoder256,
    MedianCutSixelEncoderMono,
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

    /// Use ADU intead of Focal for palette generation.
    #[clap(long, short = 'f', default_value_t = PaletteFormat::Focal)]
    palette_format: PaletteFormat,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let image_path = args.image_path;

    let image = image::load_from_memory(&read(image_path)?)?;
    let image = image.to_rgb8();

    let six = match args.palette_format {
        PaletteFormat::Adu => match args.palette_size {
            0..12 => <ADUSixelEncoder8>::encode(image),
            12..24 => <ADUSixelEncoder16>::encode(image),
            24..48 => <ADUSixelEncoder32>::encode(image),
            48..86 => <ADUSixelEncoder64>::encode(image),
            86..192 => <ADUSixelEncoder128>::encode(image),
            _ => <ADUSixelEncoder256>::encode(image),
        },
        PaletteFormat::Focal => match args.palette_size {
            0..3 => <FocalSixelEncoderMono>::encode(image),
            3..6 => <FocalSixelEncoder4>::encode(image),
            6..12 => <FocalSixelEncoder8>::encode(image),
            12..24 => <FocalSixelEncoder16>::encode(image),
            24..48 => <FocalSixelEncoder32>::encode(image),
            48..86 => <FocalSixelEncoder64>::encode(image),
            86..192 => <FocalSixelEncoder128>::encode(image),
            _ => <FocalSixelEncoder256>::encode(image),
        },
        PaletteFormat::MedianCut => match args.palette_size {
            0..3 => <MedianCutSixelEncoderMono>::encode(image),
            3..6 => <MedianCutSixelEncoder4>::encode(image),
            6..12 => <MedianCutSixelEncoder8>::encode(image),
            12..24 => <MedianCutSixelEncoder16>::encode(image),
            24..48 => <MedianCutSixelEncoder32>::encode(image),
            48..86 => <MedianCutSixelEncoder64>::encode(image),
            86..192 => <MedianCutSixelEncoder128>::encode(image),
            _ => <MedianCutSixelEncoder256>::encode(image),
        },
    };

    println!("{six}");

    Ok(())
}
