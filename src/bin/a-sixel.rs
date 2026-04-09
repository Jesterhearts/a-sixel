use std::fs::read;
use std::io::Cursor;
use std::io::IsTerminal;
use std::io::Read;
use std::io::stdin;

use a_sixel::PaletteBuilder;
use a_sixel::SixelEncoder;
use a_sixel::dither::Dither;
use clap::CommandFactory;
use clap::Parser;
use image::ImageReader;
use image::RgbaImage;

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
    #[clap(long, short = 'a', default_value_t = PaletteBuilder::Bit)]
    algorithm: PaletteBuilder,

    /// The dithering algorithm to use.
    #[clap(long, short, default_value_t = Dither::None)]
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

    let encoder = SixelEncoder::new(args.algorithm, args.dither);

    for image in split_images(&image_data)? {
        let six = encoder.encode_with_palette_size(image, args.palette_size);
        println!("{six}");
    }

    Ok(())
}
