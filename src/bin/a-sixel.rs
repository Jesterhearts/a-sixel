use std::fs::read;
use std::io::Cursor;
use std::io::IsTerminal;
use std::io::Read;
use std::io::stdin;

use a_sixel::ADUSixelEncoder;
use a_sixel::BitMergeSixelEncoder;
use a_sixel::BitMergeSixelEncoderBest;
use a_sixel::BitMergeSixelEncoderBetter;
use a_sixel::BitMergeSixelEncoderLow;
use a_sixel::BitSixelEncoder;
use a_sixel::FocalSixelEncoder;
use a_sixel::KMeansSixelEncoder;
use a_sixel::KMediansSixelEncoder;
use a_sixel::MedianCutSixelEncoder;
use a_sixel::OctreeSixelEncoder;
use a_sixel::WuSixelEncoder;
use a_sixel::dither::Bayer;
use a_sixel::dither::NoDither;
use a_sixel::dither::Sobol;
use clap::CommandFactory;
use clap::Parser;
use image::ImageReader;
use image::RgbaImage;
use strum::Display;
use strum::EnumString;

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumString, Display)]
#[strum(ascii_case_insensitive, serialize_all = "kebab-case")]
enum Algorithm {
    Adu,
    Bit,
    BitMergeLow,
    BitMerge,
    BitMergeBetter,
    BitMergeBest,
    Focal,
    #[strum(serialize = "kmeans", serialize = "k-means")]
    KMeans,
    #[strum(
        serialize = "kmedians",
        serialize = "kmed",
        serialize = "k-medians",
        serialize = "k-med"
    )]
    KMedians,
    MedianCut,
    Octree,
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
#[command(name = "a-sixel", about = "Encode images as sixel graphics")]
struct Args {
    /// The path to the image file to be loaded and rendered.
    ///
    /// At least one of this positional argument or -i/--image-path is required
    /// unless you are piping from stdin e.g. via `cat image.png | a-sixel`
    positional_image_paths: Vec<String>,

    /// The path to the image file to be loaded and rendered.
    #[clap(long = "image_path", short = 'i')]
    specified_image_paths: Vec<String>,

    /// The palette size to use for quantization, will be rounded to the nearest
    /// power of 2.
    #[clap(long, short, default_value_t = 256)]
    palette_size: usize,

    /// The palette generator to use.
    #[clap(long, short = 'a', default_value_t = Algorithm::Bit)]
    algorithm: Algorithm,

    /// The dithering algorithm to use.
    #[clap(long, short, default_value_t = Dither::No)]
    dither: Dither,
}

fn split_images(data: &[u8]) -> anyhow::Result<Vec<RgbaImage>> {
    let mut images = vec![];
    let mut cursor = Cursor::new(data);
    while cursor.position() < data.len() as u64 {
        let Ok(image) = ImageReader::new(&mut cursor)
            .with_guessed_format()?
            .decode()
        else {
            // This is a really stupid way to ensure we handle all the different possible
            // eof markers, since image doesn't advance the cursor past them. If
            // someone does something silly like pass in a huge invalid image
            // buffer, it'll be slowish, but at that point who cares.
            let pos = cursor.position();
            cursor.set_position(pos.saturating_add(1));
            continue;
        };
        images.push(image.to_rgba8());
    }

    Ok(images)
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let image_data;
    if !stdin().is_terminal() {
        let mut buf = vec![];
        stdin().read_to_end(&mut buf)?;
        image_data = buf;
    } else if args.specified_image_paths.is_empty() && args.positional_image_paths.is_empty() {
        Args::command().print_help()?;
        return Ok(());
    } else {
        let mut buf = vec![];
        for path in args.specified_image_paths {
            buf.extend(read(path)?);
        }
        for path in args.positional_image_paths {
            buf.extend(read(path)?);
        }
        image_data = buf;
    }

    for image in split_images(&image_data)? {
        let six = match args.algorithm {
            Algorithm::Adu => match args.dither {
                Dither::No => {
                    <ADUSixelEncoder<NoDither>>::encode_with_palette_size(image, args.palette_size)
                }
                Dither::Sobol => {
                    <ADUSixelEncoder<Sobol>>::encode_with_palette_size(image, args.palette_size)
                }
                Dither::Sierra => {
                    <ADUSixelEncoder>::encode_with_palette_size(image, args.palette_size)
                }
                Dither::Bayer => {
                    <ADUSixelEncoder<Bayer>>::encode_with_palette_size(image, args.palette_size)
                }
            },
            Algorithm::Bit => match args.dither {
                Dither::No => {
                    <BitSixelEncoder<NoDither>>::encode_with_palette_size(image, args.palette_size)
                }
                Dither::Sobol => {
                    <BitSixelEncoder<Sobol>>::encode_with_palette_size(image, args.palette_size)
                }
                Dither::Sierra => {
                    <BitSixelEncoder>::encode_with_palette_size(image, args.palette_size)
                }
                Dither::Bayer => {
                    <BitSixelEncoder<Bayer>>::encode_with_palette_size(image, args.palette_size)
                }
            },
            Algorithm::BitMergeLow => match args.dither {
                Dither::No => <BitMergeSixelEncoderLow<NoDither>>::encode_with_palette_size(
                    image,
                    args.palette_size,
                ),
                Dither::Sobol => <BitMergeSixelEncoderLow<Sobol>>::encode_with_palette_size(
                    image,
                    args.palette_size,
                ),
                Dither::Sierra => {
                    <BitMergeSixelEncoderLow>::encode_with_palette_size(image, args.palette_size)
                }
                Dither::Bayer => <BitMergeSixelEncoderLow<Bayer>>::encode_with_palette_size(
                    image,
                    args.palette_size,
                ),
            },
            Algorithm::BitMerge => match args.dither {
                Dither::No => <BitMergeSixelEncoder<NoDither>>::encode_with_palette_size(
                    image,
                    args.palette_size,
                ),
                Dither::Sobol => <BitMergeSixelEncoder<Sobol>>::encode_with_palette_size(
                    image,
                    args.palette_size,
                ),
                Dither::Sierra => {
                    <BitMergeSixelEncoder>::encode_with_palette_size(image, args.palette_size)
                }
                Dither::Bayer => <BitMergeSixelEncoder<Bayer>>::encode_with_palette_size(
                    image,
                    args.palette_size,
                ),
            },
            Algorithm::BitMergeBetter => match args.dither {
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
            Algorithm::BitMergeBest => match args.dither {
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
            Algorithm::Focal => match args.dither {
                Dither::No => <FocalSixelEncoder<NoDither>>::encode_with_palette_size(
                    image,
                    args.palette_size,
                ),
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
            Algorithm::KMeans => match args.dither {
                Dither::No => <KMeansSixelEncoder<NoDither>>::encode_with_palette_size(
                    image,
                    args.palette_size,
                ),
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
            Algorithm::KMedians => match args.dither {
                Dither::No => <KMediansSixelEncoder<NoDither>>::encode_with_palette_size(
                    image,
                    args.palette_size,
                ),
                Dither::Sobol => <KMediansSixelEncoder<Sobol>>::encode_with_palette_size(
                    image,
                    args.palette_size,
                ),
                Dither::Sierra => {
                    <KMediansSixelEncoder>::encode_with_palette_size(image, args.palette_size)
                }
                Dither::Bayer => <KMediansSixelEncoder<Bayer>>::encode_with_palette_size(
                    image,
                    args.palette_size,
                ),
            },
            Algorithm::MedianCut => match args.dither {
                Dither::No => <MedianCutSixelEncoder<NoDither>>::encode_with_palette_size(
                    image,
                    args.palette_size,
                ),
                Dither::Sobol => <MedianCutSixelEncoder<Sobol>>::encode_with_palette_size(
                    image,
                    args.palette_size,
                ),
                Dither::Sierra => {
                    <MedianCutSixelEncoder>::encode_with_palette_size(image, args.palette_size)
                }
                Dither::Bayer => <MedianCutSixelEncoder<Bayer>>::encode_with_palette_size(
                    image,
                    args.palette_size,
                ),
            },
            Algorithm::Octree => match args.dither {
                Dither::No => <OctreeSixelEncoder<NoDither>>::encode_with_palette_size(
                    image,
                    args.palette_size,
                ),
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
            Algorithm::Wu => match args.dither {
                Dither::No => {
                    <WuSixelEncoder<NoDither>>::encode_with_palette_size(image, args.palette_size)
                }
                Dither::Sobol => {
                    <WuSixelEncoder<Sobol>>::encode_with_palette_size(image, args.palette_size)
                }
                Dither::Sierra => {
                    <WuSixelEncoder>::encode_with_palette_size(image, args.palette_size)
                }
                Dither::Bayer => {
                    <WuSixelEncoder<Bayer>>::encode_with_palette_size(image, args.palette_size)
                }
            },
        };

        println!("{six}");
    }

    Ok(())
}
